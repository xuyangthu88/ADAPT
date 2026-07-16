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
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

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

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps):
        print('start train diffusion ......')
        timer = Timer()
        loss_list = []
        recon_list = []
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)

                # update_trajectory = torch.concat((batch[0], batch[2]))
                # update_condition = torch.concat((batch[1][0], batch[3][0]))
                # update_trajectory = batch[2]
                # update_condition = batch[3][0]
                # update_condition = {0: update_condition}
                # batch = UpdateBatch(update_trajectory, update_condition)
                # print(self.dataset.normalizer)
                update_batch = batch_to_device(batch)
                loss, infos, recon_dist = self.model.loss(self.action_means, self.action_stds, *update_batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                loss_list.append(loss.item())
                # recon_dist = self.compute_recon_dist(x_recon, x_start)
                recon_list.append(recon_dist.item())

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                self.save(label)

            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.4f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.4f} | {infos_str} | t: {timer():8.4f}', flush=True)
                print('horizon: ', Horizon, ' dim: ', Dim, ' n diffusion step: ', n_diffusion_steps)
                print("compute state dist:", recon_dist.item())
            self.step += 1

        train_log = {
            'diffusion_loss_mean': np.mean(loss_list),
            'diffusion_loss_std': np.std(loss_list),
            'diffusion_loss_decrease': loss_list[0] - loss_list[-1],
            'recon_dist_mean': np.mean(recon_list),
            'recon_dist_std': np.std(recon_list)

            # 'diffusion_loss': loss.item()
        }
        print('end train diffusion ......')
        return train_log

    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
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