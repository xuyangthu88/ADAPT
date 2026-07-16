from collections import namedtuple
import torch

import diffuser.utils as utils
from diffuser.datasets.preprocessing import get_policy_preprocess_fn

DEVISE = None

Trajectories = namedtuple('Trajectories', 'actions observations values')


class GuidedPolicy:

    def __init__(self, diffusion_model, normalizer, preprocess_fns, 
                    **sample_kwargs):
        self.diffusion_model = diffusion_model
        # diffusion_model: gaussian diffusion
        self.normalizer = normalizer
        self.action_dim = diffusion_model.action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.sample_kwargs = sample_kwargs
    
    def __call__(self, conditions, batch_size=1, verbose=True):
        conditions = self._format_conditions(conditions, batch_size)

        ## run reverse diffusion process
        # diffusion_model: gaussian diffusion
        samples = self.diffusion_model(conditions, verbose=verbose, **self.sample_kwargs)
        trajectories = utils.to_np(samples.trajectories)
        
        ## extract action [ batch_size x horizon x transition_dim ]
        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        
        action = actions[:, 0]

        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')
        
        trajectories = Trajectories(actions, observations, samples.values)
        return action, trajectories

    def get_action_only(self, conditions, batch_size=1, verbose=True):
        """
        简化的调用函数，只返回动作而不返回完整的轨迹

        参数:
            conditions: 条件输入
            batch_size: 批量大小，默认为1
            verbose: 是否显示详细信息，默认为True

        返回:
            action: 第一个时间步的动作 [batch_size x action_dim]
        """
        conditions = self._format_conditions(conditions, batch_size)

        # 运行反向扩散过程
        samples = self.diffusion_model.sample_with_grad(conditions, verbose=verbose, **self.sample_kwargs)

        # 提取动作 [batch_size x horizon x action_dim]
        actions = samples.trajectories[:, :, :self.action_dim]

        # 反规范化动作
        actions = self.normalizer.unnormalize(actions, 'actions')

        # 返回第一个时间步的动作
        return actions[:, 0]

    @property
    def device(self):
        parameters = list(self.diffusion_model.parameters())
        return parameters[0].device

    def _format_conditions(self, conditions, batch_size):
        conditions = utils.apply_dict(
            self.normalizer.normalize,
            conditions,
            'observations',
        )
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=DEVISE)
        return conditions
