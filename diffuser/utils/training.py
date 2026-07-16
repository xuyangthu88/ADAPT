import os
import copy
import numpy as np
import torch
import einops
import pdb

from .arrays import batch_to_device, to_np, to_device, apply_dict, normalizer_to_device
from .timer import Timer
from .cloud import sync_logs

from collections import namedtuple
UpdateBatch = namedtuple('Batch', 'trajectories conditions')

# change for z
from diffuser.z_model.encoders import EncoderP, EncoderQ

Dim = None
Horizon = None
n_diffusion_steps = None

DEVISE = None

def cycle(dl):
    while True:
        for data in dl:
            yield data

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def reparameterize(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + std * eps

def kl_normal(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (logvar_p - logvar_q + (var_q + (mu_q - mu_p).pow(2)) / var_p - 1.0)
    free_bits = 0.01  # 每维最小 KL
    kl = torch.clamp(kl, min=free_bits)
    return kl.sum(dim=-1)  # sum over z dim

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        use_latent_encoder=True,
        z_dim=16,
        constant_z_value=0.0,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.use_latent_encoder = use_latent_encoder
        self.z_dim = z_dim
        self.constant_z_value = constant_z_value


        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.train_lr = train_lr
        # change for z
        self.building_offline_models()

        self.reset_parameters()
        self.step = 0
        # self.normalizer = normalizer_to_device(self.dataset.normalizer)
        self.normalizer = self.dataset.normalizer
        # print(self.normalizer["actions"])
        self.action_means = normalizer_to_device(torch.Tensor(self.normalizer.means['actions']))
        self.action_stds = normalizer_to_device(torch.Tensor(self.normalizer.stds['actions']))

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)
        if self.use_latent_encoder:
            self.ema.update_model_average(self.ema_model_p, self.model_p)
            self.ema.update_model_average(self.ema_model_q, self.model_q)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#
    # change for z
    def building_offline_models(self):
        state_dim = self.dataset.action_dim
        obs_dim = self.dataset.observation_dim
        z_dim = self.z_dim
        if not self.use_latent_encoder:
            self.model_p = None
            self.model_q = None
            self.opt_p = None
            self.opt_q = None
            self.ema_model_p = None
            self.ema_model_q = None
            print(f'[ utils/training ] Latent encoders disabled; using constant z_dim={z_dim}')
            return
        device = next(self.model.parameters()).device
        self.model_p = EncoderP(x_dim=obs_dim, z_dim=z_dim, hidden=1024).to(device)
        self.model_q = EncoderQ(x_dim=obs_dim, y_dim=state_dim, z_dim=z_dim, hidden=1024).to(device)
        self.opt_p = torch.optim.AdamW(self.model_p.parameters(), lr=self.train_lr, weight_decay=1e-4)
        self.opt_q = torch.optim.AdamW(self.model_q.parameters(), lr=self.train_lr, weight_decay=1e-4)
        print(self.model_p)
        print(self.model_q)
        # self.beta_KL = 0.1
        self.beta_KL_offline = 1 # loss = loss_diff + beta_KL_offline*loss_kl
        self.ema_model_p = copy.deepcopy(self.model_p)
        self.ema_model_q = copy.deepcopy(self.model_q)

    def train(self, n_train_steps):
        print('start train diffusion ......')
        timer = Timer()
        loss_list = []
        loss_diff_list = []
        loss_kl_list = []
        recon_list = []
        info_lists = {}
        warmup_steps = 10000  # KL warm-up 步数
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                # update_batch = batch_to_device(batch)
                batch = batch_to_device(batch)
                # change for z
                B = batch.trajectories.size(0)
                q_inpt = batch.trajectories.view(B, -1) # state+obs
                p_inpt = batch.conditions[0]
                if self.use_latent_encoder:
                    # posterior q(z|x,y)
                    mu_q, logvar_q = self.model_q(q_inpt)
                    # prior p(z|x)
                    mu_p, logvar_p = self.model_p(p_inpt)
                    z_q = reparameterize(mu_q, logvar_q)
                else:
                    z_q = torch.full(
                        (B, self.z_dim),
                        self.constant_z_value,
                        dtype=batch.trajectories.dtype,
                        device=batch.trajectories.device,
                    )
                horizon = batch.trajectories.size(1)
                z_seq = z_q.unsqueeze(1).expand(B, horizon, self.z_dim)
                x_cat = torch.cat([batch.trajectories, z_seq], dim=-1)  # [B, H, state+obs+z_dim]
                cond_cat =  torch.cat([p_inpt, z_q], dim=-1) # [B, 186+z_dim]

                # 构造新的 conditions
                new_conditions = {key: cond_cat for key in batch.conditions.keys()}

                # 用 Batch 类重新构造
                update_batch = batch.__class__(trajectories=x_cat, conditions=new_conditions)
                # update_batch = batch_to_device(batch)
                # update_trajectory = torch.concat((batch[0], batch[2]))
                # update_condition = torch.concat((batch[1][0], batch[3][0]))
                # update_trajectory = batch[2]
                # update_condition = batch[3][0]
                # update_condition = {0: update_condition}
                # batch = UpdateBatch(update_trajectory, update_condition)
                # print(self.dataset.normalizer)

                loss_diff, infos, recon_dist= self.model.loss(self.action_means, self.action_stds, *update_batch)
                if self.use_latent_encoder:
                    loss_kl = kl_normal(mu_q, logvar_q, mu_p, logvar_p).mean()
                else:
                    loss_kl = torch.zeros((), dtype=loss_diff.dtype, device=loss_diff.device)
                # loss = loss_diff + self.beta_KL_offline * loss_kl
                beta = min(1.0, self.step / warmup_steps)
                loss = loss_diff + beta * loss_kl if self.use_latent_encoder else loss_diff
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                loss_list.append(loss.item())
                loss_diff_list.append(loss_diff.item())
                loss_kl_list.append(loss_kl.item())
                # recon_dist = self.compute_recon_dist(x_recon, x_start)
                recon_list.append(recon_dist.item())
                for key, val in infos.items():
                    if torch.is_tensor(val):
                        val = val.detach().mean().item()
                    else:
                        val = float(val)
                    info_lists.setdefault(key, []).append(val)
            
            self.optimizer.step()
            # change for z
            if self.use_latent_encoder:
                torch.nn.utils.clip_grad_norm_(self.model_p.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.model_q.parameters(), 1.0)
                self.opt_p.step()
                self.opt_q.step()
            self.optimizer.zero_grad()
            # change for z
            if self.use_latent_encoder:
                self.opt_p.zero_grad()
                self.opt_q.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)
            
            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                print(f'loss_diff: {loss_diff:8.4f} | loss_kl: {loss_kl:8.4f}', flush=True)
                print('horizon: ', Horizon, ' dim: ', Dim, ' n diffusion step: ', n_diffusion_steps)
                print("compute state dist:", recon_dist.item())
            self.step += 1

        train_log = {
            'diffusion_loss_mean': np.mean(loss_list),
            'diffusion_loss_std': np.std(loss_list),
            'diffusion_loss_decrease': loss_list[0] - loss_list[-1],
            'diffusion_loss_diff_mean': np.mean(loss_diff_list),
            'diffusion_loss_diff_std': np.std(loss_diff_list),
            'diffusion_loss_kl_mean': np.mean(loss_kl_list),
            'diffusion_loss_kl_std': np.std(loss_kl_list),
            'recon_dist_mean': np.mean(recon_list),
            'recon_dist_std': np.std(recon_list)

            # 'diffusion_loss': loss.item()
        }
        for key, values in info_lists.items():
            train_log[f'{key}_mean'] = np.mean(values)
            train_log[f'{key}_std'] = np.std(values)
        print('end train diffusion ......')
        return train_log

    # def compute_z(self):


    def kl_loss(self):
        # prior p(z|x)

        return None

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
            'use_latent_encoder': self.use_latent_encoder,
            'z_dim': self.z_dim,
            'constant_z_value': self.constant_z_value,
        }
        if self.use_latent_encoder:
            data.update({
                'model_p': self.model_p.state_dict(),
                'model_p_ema': self.ema_model_p.state_dict(),
                'model_q': self.model_q.state_dict(),
            })
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, map_location=DEVISE)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])
