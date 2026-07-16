from collections import namedtuple
import numpy as np
import torch
from torch import nn
import pdb

import diffuser.utils as utils
from .helpers import (
    cosine_beta_schedule,
    extract,
    apply_conditioning,
    Losses, apply_conditioning_with_grad, n_step_guided_p_sample_with_grad
)


Sample = namedtuple('Sample', 'trajectories values chains')

@torch.no_grad()
def default_sample_fn(model, x, cond, t):
    model_mean, _, model_log_variance = model.p_mean_variance(x=x, cond=cond, t=t)
    model_std = torch.exp(0.5 * model_log_variance)

    # no noise when t == 0
    noise = torch.randn_like(x)
    noise[t == 0] = 0

    values = torch.zeros(len(x), device=x.device)
    return model_mean + model_std * noise, values


def sort_by_values(x, values):
    inds = torch.argsort(values, descending=True)
    x = x[inds]
    values = values[inds]
    return x, values


def make_timesteps(batch_size, i, device):
    t = torch.full((batch_size,), i, device=device, dtype=torch.long)
    return t


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, n_timesteps=1000,
        loss_type='l1', clip_denoised=False, predict_epsilon=True,
        action_weight=1.0, loss_discount=1.0, loss_weights=None,
        action_loss_weights=None, smoothl1_beta=1.0,
    ):
        super().__init__()

        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.transition_dim = observation_dim + action_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.smoothl1_beta = smoothl1_beta

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
            torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        ## get loss coefficients and initialize objective
        loss_weights = self.get_loss_weights(action_weight, loss_discount, loss_weights, action_loss_weights)
        self.loss_fn = Losses[loss_type](loss_weights, self.action_dim, smoothl1_beta=smoothl1_beta)

        self.loss_type = loss_type
        self.physics_loss = None

        ## compute state dist
        self.compute_state_dist = True

    def get_loss_weights(self, action_weight, discount, weights_dict, action_weights_dict=None):
        '''
            sets loss coefficients for trajectory

            action_weight   : float
                coefficient on first action loss
            discount   : float
                multiplies t^th timestep of trajectory loss by discount**t
            weights_dict    : dict
                { i: c } multiplies dimension i of observation loss by c
        '''
        self.action_weight = action_weight

        dim_weights = torch.ones(self.transition_dim, dtype=torch.float32)

        ## set loss coefficients for dimensions of observation
        if weights_dict is None: weights_dict = {}
        for ind, w in weights_dict.items():
            dim_weights[self.action_dim + ind] *= w

        ## set loss coefficients for dimensions of action / predicted targets
        if action_weights_dict is None: action_weights_dict = {}
        for ind, w in action_weights_dict.items():
            dim_weights[ind] *= w

        ## decay loss with trajectory timestep: discount**t
        discounts = discount ** torch.arange(self.horizon, dtype=torch.float)
        discounts = discounts / discounts.mean()
        loss_weights = torch.einsum('h,t->ht', discounts, dim_weights)

        ## weight all predicted target steps; for horizon=1 this matches the original a0 weighting
        loss_weights[:, :self.action_dim] = action_weight * dim_weights[:self.action_dim]
        return loss_weights

    #------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        # self.predict_epsilon: False
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, 
                            t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, 
                            t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        # self.model: TemporalUnet
        # predict_start_from_noise: 输入noise, cond, t, 直接输出模型的值
        x_recon = self.predict_start_from_noise(x, 
                        t=t, noise=self.model(x, cond, t))
        
        if self.clip_denoised:
            x_recon.clamp_(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_loop(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn, **sample_kwargs):
        device = self.betas.device
        
        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        # print(x.shape, cond.shape, "222222")
        x = apply_conditioning(x, cond, self.action_dim)
        # x: (64, 4, 23) (随机噪声，加入了horizon); cond: [(64, 17)] (cond: 观测重复64次)
        # horizon为时间拍，用 obs 替换 t=0 的随机噪声

        chain = [x] if return_chain else None
        
        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # self.n_timesteps: 20; self: Gaussian Diffusion
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            # t: (64), t = 20, ..., 1
            x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            # print(x.shape,"1111")
            x = apply_conditioning(x, cond, self.action_dim)
        
        return Sample(x, values, chain)

    @torch.no_grad()
    def conditional_sample(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop(shape, cond, **sample_kwargs)
    
    #------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        # print('extract: ', self.sqrt_alphas_cumprod.shape, 
            # extract(self.sqrt_alphas_cumprod, t, x_start.shape).shape)
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, action_means, action_stds, x_start, cond, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print('x_start: ', x_start.shape, ' cond: ', cond[0].shape, ' t: ', t.shape, ' n_time: ', self.n_timesteps)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)
        
        x_recon = self.model(x_noisy, cond, t)
        # self.model: TemporalUnet
        # print('x_recon: ', x_recon.shape)
        x_recon = apply_conditioning(x_recon, cond, self.action_dim)
        # print(x_recon.shape, noise.shape, x_noisy.shape, x_noisy.shape)
        # breakpoint()
        # print('='*10)
        # print(x_noisy[0].shape)
        # print(x_recon[0].shape)

        assert noise.shape == x_recon.shape
        
        if self.predict_epsilon:
            loss, info, loss_detach = self.loss_fn(x_recon, noise)
            x0_for_physics = self.predict_start_from_noise(x_noisy, t=t, noise=x_recon)
        else:
            loss, info, loss_detach = self.loss_fn(x_recon, x_start)
            x0_for_physics = x_recon

        if self.physics_loss is not None:
            physics_loss, physics_info = self.physics_loss(x0_for_physics, cond, x_start=x_start)
            loss = loss + physics_loss
            info.update(physics_info)

        # loss_detach = torch.sum(loss_detach, dim=0).view(-1)
        # loss_detach_action = loss_detach[:self.action_dim]
        # loss_detach_action_prob = (loss_detach_action / torch.sum(loss_detach_action) * len(loss_detach_action) * 50).detach().cpu()
        # obs_weight = torch.ones(self.transition_dim-self.action_dim)
        # loss_detach_action_prob = torch.concat([loss_detach_action_prob, obs_weight]).view(1, -1)
        # self.loss_fn = Losses[self.loss_type](loss_detach_action_prob, 
        #                                             self.action_dim)
        
        # if self.compute_state_dist:
        state_recon, state_original = self.compute_recon_dist(x_recon, x_start, action_means, action_stds)
        
        # recon_dist = torch.nn.functional.mse_loss(state_recon, state_original)
        recon_dist = torch.abs(state_recon - state_original) # / (torch.abs(state_original) + 1e-6)
        recon_dist = torch.sum(recon_dist, dim=-1)
        recon_dist = torch.mean(recon_dist)

        return loss, info, recon_dist

    def loss(self, action_means, action_stds, x, *args):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        # print('batch size: ', x.shape, '  t: ', t.shape, ' timestep: ', self.n_timesteps)
        return self.p_losses(action_means, action_stds, x, *args, t)
        
    def compute_recon_dist(self, x_recon, x_original, action_means, action_stds):
        norm_recon = x_recon[:, :, :self.action_dim]
        norm_original = x_original[:, :, :self.action_dim]
        state_recon = action_means + norm_recon * action_stds
        state_original = action_means + norm_original * action_stds

        return state_recon, state_original

    def p_sample_loop_with_grad(self, shape, cond, verbose=True, return_chain=False, sample_fn=default_sample_fn,
                      **sample_kwargs):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)
        x = apply_conditioning_with_grad(x, cond, self.action_dim)
        # x: (64, 4, 23) (随机噪声，加入了horizon); cond: [(64, 17)] (cond: 观测重复64次)
        # horizon为时间拍，用 obs 替换 t=0 的随机噪声

        chain = [x] if return_chain else None

        # progress = utils.Progress(self.n_timesteps) if verbose else utils.Silent()
        # self.n_timesteps: 20; self: Gaussian Diffusion
        for i in reversed(range(0, self.n_timesteps)):
            t = make_timesteps(batch_size, i, device)
            # t: (64), t = 20, ..., 1
            # x, values = sample_fn(self, x, cond, t, **sample_kwargs)
            x, values = n_step_guided_p_sample_with_grad(self, x, cond, t, **sample_kwargs)
            x = apply_conditioning_with_grad(x, cond, self.action_dim)

        return Sample(x, values, chain)

    def conditional_sample_with_grad(self, cond, horizon=None, **sample_kwargs):
        '''
            conditions : [ (time, state), ... ]
        '''
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = horizon or self.horizon
        shape = (batch_size, horizon, self.transition_dim)

        return self.p_sample_loop_with_grad(shape, cond, **sample_kwargs)

    def forward(self, cond, *args, **kwargs):
        return self.conditional_sample(cond, *args, **kwargs)

    def sample_with_grad(self, cond, *args, ** kwargs):
        return self.conditional_sample_with_grad(cond, *args, **kwargs)

class ValueDiffusion(GaussianDiffusion):

    def p_losses(self, x_start, cond, target, t):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # print('x_start: ', x_start.shape, ' t: ', t.shape, ' n_time: ', self.n_timesteps)
        x_noisy = apply_conditioning(x_noisy, cond, self.action_dim)

        pred = self.model(x_noisy, cond, t)
        # self.model: ValueFunction
        # print('pred: ', pred.shape, ' target: ', target.shape, ' x_noisy:', x_noisy.shape)
        loss, info = self.loss_fn(pred, target)
        return loss, info

    def forward(self, x, cond, t):
        return self.model(x, cond, t)
