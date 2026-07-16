import diffuser.sampling as sampling
import diffuser.utils as utils
import algos

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffuser.utils.arrays import batch_to_device

utils.serialization.DEVISE = 'cuda:0'
sampling.DEVISE = 'cuda:0'
utils.training.DEVISE = 'cuda:0'
utils.arrays.DEVICE = 'cuda:0'
algos.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class Neural(nn.Module):
	def __init__(self, input_dim, action_dim):
		super(Neural, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(input_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)


	def forward(self, input_obs):
		q1 = F.relu(self.l1(input_obs))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)
		return q1


class MLP(object):
    def __init__(self, input_dim, action_dim):
        self.neural = Neural(input_dim, action_dim).to(device)
        self.neural_optimizer = torch.optim.Adam(self.neural.parameters(), lr=3e-4)

    def train(self, input_state, target):
        predict = self.neural(input_state)
        neural_loss = F.mse_loss(predict, target)

        self.neural_optimizer.zero_grad()
        neural_loss.backward()
        self.neural_optimizer.step()

        return neural_loss.detach().cpu().numpy()

    def predict(self, input_state):
        return self.neural(input_state)

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    # dataset: str = 'walker2d-medium-replay-v2'
    dataset: str = "10gen_zerg10v11"
    # MMM2
    # 3s5z_vs_3s6z
    # 6h_vs_8z
    # 10gen_terran10v11 10gen_terran5v5
    # 10gen_protoss5v5
    # 10gen_zerg10v10 10gen_zerg10v11
    # 10gen_zerg5v5
    logbase: str = "logs"
    config: str = 'config.locomotion'
    # pretrained_data_path = '/home/yyq/diffuser-v3light'
    pretrained_data_path = '/home/yyq-shixi3/observation/diffuser-light_train'

args = Parser().parse_args('plan', args=[
    # "--logbase", "/home/yyq/diffuser-v3light/logs",
    # "--loadbase", "/home/yyq/diffuser-v3light/logs",
    "--logbase", "/home/yyq-shixi3/observation/diffuser-light_train/logs",
    "--loadbase", "/home/yyq-shixi3/observation/diffuser-light_train/logs",
    ])


argst = Parser().parse_args('diffusion')

utils.training.Dim = str(argst.dim_mults)
utils.training.Horizon = str(argst.horizon)
utils.training.n_diffusion_steps = str(argst.n_diffusion_steps)

# change for z
utils.training_online.Dim = str(argst.dim_mults)
utils.training_online.Horizon = str(argst.horizon)
utils.training_online.n_diffusion_steps = str(argst.n_diffusion_steps)
#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

import numpy as np
import os
from diffuser.z_model.encoders import EncoderP

base_dir = "/home/yyq-shixi3/observation/diffuser-light_train/logs"
## load diffusion model and value function from disk
load_path = 'MA_H_' + str(args.horizon) + '_Dim_' + str(args.dim_mults) + '_T_' + str(args.n_diffusion_steps) + '_ratio_' + str(args.data_ratio) + '_' + str(args.dataset) + '_' + str(args.loss_type) + '_new'
# load_path = 'MA_H_' + str(args.horizon) + '_Dim_' + str(args.dim_mults) + '_T_' + str(args.n_diffusion_steps) + '_ratio_' + str(args.data_ratio) + '_' + str(args.dataset) + '_' + str(args.loss_type) + '_r1'

diffusion_experiment = utils.load_diffusion(
    args.loadbase, load_path, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
# online
# diffusion_experiment = utils.load_diffusion(
#     args.loadbase, load_path, args.diffusion_loadpath,
#     epoch=0, seed=args.seed,
# )

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset
# change for z
z_dim = 16
z_means = np.zeros(z_dim)
z_stds  = np.ones(z_dim)
# obs_means = dataset.normalizer.means["observations"]
# obs_stds = dataset.normalizer.stds["observations"]
# dataset.normalizer.normalizers["observations"].means = np.concatenate([obs_means, z_means])
# dataset.normalizer.normalizers["observations"].stds  = np.concatenate([obs_stds,  z_stds])
import copy
# 创建完全独立的拷贝
policy_normalizer = copy.deepcopy(dataset.normalizer)
obs_means = policy_normalizer.means["observations"]
obs_stds = policy_normalizer.stds["observations"]
policy_normalizer.normalizers["observations"].means = np.concatenate([obs_means, z_means])
policy_normalizer.normalizers["observations"].stds  = np.concatenate([obs_stds,  z_stds])

# breakpoint()
# diff_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
# print(f"diffusion 参数总数: {diff_params:,}")

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    scale=args.scale,
    diffusion_model=diffusion,
    # normalizer=dataset.normalizer,
    normalizer=policy_normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
policy = policy_config()
# print(policy)

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
# change for z
def get_pretrained_p_net():
    obs_dim = dataset.observation_dim
    # print("obs_dim", obs_dim)
    z_dim = 16
    model_p = EncoderP(obs_dim, z_dim, hidden=1024).to(device)
    ema_model_p = EncoderP(obs_dim, z_dim, hidden=1024).to(device)
    ckpt_file = os.path.join(
        base_dir,
        load_path,
        f"diffusion/defaults_H1_T{args.n_diffusion_steps}/state_800000.pt"
    )
    # online
    # ckpt_file = os.path.join(
    #     base_dir,
    #     load_path,
    #     f"diffusion/defaults_H1_T{args.n_diffusion_steps}/state_0.pt"
    # )
    checkpoint = torch.load(ckpt_file, map_location=device)
    model_p.load_state_dict(checkpoint["model_p"])
    # model_p.load_state_dict(checkpoint["model_p_ema"])
    # ema_model_p.load_state_dict(checkpoint["model_p_ema"])
    ema_model_p = copy.deepcopy(model_p)
    print(f"[Load] Loaded model_p & model_p_ema from {ckpt_file}")
    # print(
    #     model_p
    # )
    # breakpoint()
    return model_p, ema_model_p

def load_visual_p(file_path):
    obs_dim = dataset.observation_dim
    # print("obs_dim", obs_dim)
    z_dim = 16
    model_p = EncoderP(obs_dim, z_dim, hidden=1024).to(device)

    checkpoint = torch.load(file_path, map_location=device)
    model_p.load_state_dict(checkpoint)
    print(f"[Load] Loaded model_p & model_p_ema from {file_path}")
    # breakpoint()
    return model_p

def get_single_diffusion_state(obs):
    bs, n_agents, dim = obs.shape
    inpt = obs
    # print('input: ', inpt.reshape(bs*n_agents, -1).shape)
    state, traj = policy({0: inpt.reshape(bs*n_agents, -1)}, batch_size=args.batch_size, verbose=args.verbose)
    # print("state shape:", state.shape)
    return state.reshape(bs,n_agents,-1) # [bs,n,dim]

def get_diffusion_state(obs, concat_his_obs):
    bs, n_agents, dim = obs.shape

    if concat_his_obs is not None:
        inpt = concat_his_obs
        # breakpoint()
    else:
        inpt = []
        for bs_i in range(bs):
            inpt_i = []
            for i in range(n_agents):
                inpt_temp = np.concatenate([obs[bs_i][i, None].flatten(),
                                            obs[bs_i][:i].flatten(),
                                            obs[bs_i][i + 1:].flatten()])
                inpt_i.append(inpt_temp)
            inpt.append(np.stack(inpt_i, axis=0))
        inpt = np.stack(inpt, axis=0)

    # print('input: ', inpt.reshape(bs*n_agents, -1).shape)
    state, traj = policy({0: inpt.reshape(bs*n_agents, -1)}, batch_size=args.batch_size, verbose=args.verbose)
    # print("state shape:", state.shape)
    return state.reshape(bs,n_agents,-1) # [bs,n,dim]

def change_normalizer():
    changed_norm = copy.deepcopy(policy.normalizer)
    np_means = policy.normalizer.normalizers["observations"].means
    np_stds = policy.normalizer.normalizers["observations"].stds
    changed_norm.normalizers["observations"].means = torch.from_numpy(np_means).to(device)
    changed_norm.normalizers["observations"].stds = torch.from_numpy(np_stds).to(device)
    np_action_means = policy.normalizer.normalizers["actions"].means
    np_action_stds = policy.normalizer.normalizers["actions"].stds
    changed_norm.normalizers["actions"].means = torch.from_numpy(np_action_means).to(device)
    changed_norm.normalizers["actions"].stds = torch.from_numpy(np_action_stds).to(device)
    policy.normalizer = changed_norm
    # print(policy.normalizer)

def unchange_normalizer():
    unchanged_norm = copy.deepcopy(policy.normalizer)
    # 原来是 torch.Tensor，需要转回 numpy
    torch_means = policy.normalizer.normalizers["observations"].means
    torch_stds = policy.normalizer.normalizers["observations"].stds

    unchanged_norm.normalizers["observations"].means = torch_means.detach().cpu().numpy().copy()
    unchanged_norm.normalizers["observations"].stds = torch_stds.detach().cpu().numpy().copy()

    torch_means = policy.normalizer.normalizers["actions"].means
    torch_stds = policy.normalizer.normalizers["actions"].stds

    unchanged_norm.normalizers["actions"].means = torch_means.detach().cpu().numpy().copy()
    unchanged_norm.normalizers["actions"].stds = torch_stds.detach().cpu().numpy().copy()

    policy.normalizer = unchanged_norm

def get_sample_state(inpt):
    # print('input: ', inpt.reshape(bs*n_agents, -1).shape)
    state = policy.get_action_only({0: inpt}, batch_size=args.batch_size, verbose=args.verbose)
    # breakpoint()
    return state # [bs,n,dim]

def get_mlp_state(obs, concat_his_obs):
    bs, n_agents, dim = obs.shape

    if concat_his_obs is not None:
        inpt = concat_his_obs
    else:
        inpt = []
        for bs_i in range(bs):
            inpt_i = []
            for i in range(n_agents):
                inpt_temp = np.concatenate([obs[bs_i][i, None].flatten(),
                                            obs[bs_i][:i].flatten(),
                                            obs[bs_i][i + 1:].flatten()])
                inpt_i.append(inpt_temp)
            inpt.append(np.stack(inpt_i, axis=0))
        inpt = np.stack(inpt, axis=0)
    conditions = utils.apply_dict(
            dataset.normalizer,
            {0: inpt.reshape(bs*n_agents, -1)},
            'observations',
        )
    inpt = torch.FloatTensor(conditions[0]).to(device)

    predict = MLPNET.predict(input_state=inpt).detach().cpu().numpy()
    norm_predict = dataset.normalizer.unnormalize(predict, 'actions')
    return norm_predict.reshape(bs,n_agents,-1) # [bs,n,dim]

def get_vae_state(obs, concat_his_obs):
    bs, n_agents, dim = obs.shape
    if concat_his_obs is not None:
        inpt = concat_his_obs
    else:
        inpt = []
        for bs_i in range(bs):
            inpt_i = []
            for i in range(n_agents):
                inpt_temp = np.concatenate([obs[bs_i][i, None].flatten(),
                                            obs[bs_i][:i].flatten(),
                                            obs[bs_i][i + 1:].flatten()])
                inpt_i.append(inpt_temp)
            inpt.append(np.stack(inpt_i, axis=0))
        inpt = np.stack(inpt, axis=0).reshape(bs * n_agents, -1)
    conditions = utils.apply_dict(
            dataset.normalizer,
            {0: inpt.reshape(bs*n_agents, -1)},
            'observations',
        )
    inpt = torch.FloatTensor(conditions[0]).to(device)

    predict = vae_trainer.vae.decode(inpt).detach().cpu().numpy()
    norm_predict = dataset.normalizer.unnormalize(predict, 'actions')
    return norm_predict.reshape(bs,n_agents,-1) # [bs,n,dim]

def update_diffusion_policy(data, model_p=None, p_opt=None):
    print('update diffusion...')
    trainer_config = utils.Config(
        # change for z
        utils.Trainer_online,
        savepath=(argst.savepath, 'trainer_config.pkl'),
        train_batch_size=argst.batch_size,
        train_lr=argst.learning_rate,
        gradient_accumulate_every=argst.gradient_accumulate_every,
        ema_decay=argst.ema_decay,
        sample_freq=argst.sample_freq,
        save_freq=argst.save_freq,
        label_freq=int(argst.n_train_steps // argst.n_saves),
        save_parallel=argst.save_parallel,
        results_folder=argst.savepath,
        bucket=argst.bucket,
        n_reference=argst.n_reference,
    )
    renderer = None
    # print('update diffusion data: ', data['state'].shape, data['obs'].shape)

    # data = {'state': np.zeros((1, 400, 5, 114)),
    #         'obs': np.zeros((1, 400, 5, 80)),
    #         'done': np.zeros((1, 400, 5, 1)),
    #         'active_mask': np.zeros((1, 400, 5, 1)),}
    dataset.add_online_data(data)
    trainer = trainer_config(diffusion, dataset, renderer)

    # change for z
    if model_p is not None:
        output_log, model_p, p_opt = trainer.train(n_train_steps=10000, model_p=model_p, p_opt=p_opt)
        # output_log = trainer.train(n_train_steps=10000, model_p=model_p)
    else:
        output_log = trainer.train(n_train_steps=10000)
    return output_log, model_p, p_opt
    # return output_log

action_dim = dataset.action_dim
# get dataloader
def get_dataloader():
    train_batch_size = 32
    dataloader = cycle(torch.utils.data.DataLoader(
        dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
    ))
    return dataloader

# change for z
def get_batch_for_z(dataloader):
    train_batch_size = 32
    batch = next(dataloader)
    batch = batch_to_device(batch)
    # state + obs
    temp = batch.trajectories.view(train_batch_size, -1)
    state = temp[:, :action_dim]
    obs = temp[:, action_dim:]

    return state, obs



input_dim = dataset.observation_dim
action_dim = dataset.action_dim
MLPNET = MLP(input_dim, action_dim)  # history concat obs
def update_mlp_policy(data):
    print('start train mlp ......')
    dataset.add_online_data(data)
    train_batch_size = 32
    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
    batch = next(dataloader)
    update_trajectory = batch[0]
    dataset_concat_obs = update_trajectory[:, :, :input_dim].detach().cpu().numpy()
    dataset_state = update_trajectory[:, :, input_dim:].detach().cpu().numpy()
    
    train_iteration = 1000
    batch_size = 1
    mlp_loss_history = []
    for iter_i in range(train_iteration):
        ind = np.random.randint(0, len(dataset_state), size=batch_size)[0]
        state_torch = torch.FloatTensor(dataset_state[ind]).to(device)
        concat_obs_torch = torch.FloatTensor(dataset_concat_obs[ind]).to(device)
        mlp_loss = MLPNET.train(concat_obs_torch, state_torch)
        mlp_loss_history.append(mlp_loss)
    
    train_log = {'mlp_loss_mean': np.mean(np.array(mlp_loss_history)),}
    print('end train mlp ......')
    return train_log

def cycle(dl):
    while True:
        for data in dl:
            yield data

latent_dim = action_dim * 2
vae_lr = 1e-4
vae_hidden_size = 750
vae_trainer = algos.VAEModule(input_dim, action_dim, latent_dim, vae_lr=vae_lr, hidden_size=vae_hidden_size)
def update_vae_policy(data):
    print('start train vae ......')
    dataset.add_online_data(data)
    train_batch_size = 32
    dataloader = cycle(torch.utils.data.DataLoader(
            dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
    batch = next(dataloader)
    update_trajectory = batch[0]
    dataset_concat_obs = update_trajectory[:, :, :input_dim].detach().cpu().numpy()
    dataset_state = update_trajectory[:, :, input_dim:].detach().cpu().numpy()

    train_iteration = 1000
    train_log = vae_trainer.train(dataset_state, dataset_concat_obs, iterations=train_iteration)
    print('end train vae ......')
    return train_log


if __name__ == '__main__':
    test_state = np.load(os.path.join(args.pretrained_data_path, 'test_state.npy'))
    test_action = np.load(os.path.join(args.pretrained_data_path, 'test_action.npy'))

    test_state = np.stack((test_state, test_state))
    conditions = {0: test_state}
    action, traj = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
    
    batch_action = test_action
    mean_abs_error_list = []
    for i in range(len(action)):
        mean_abs_error = np.mean(np.abs(action[i] - batch_action))
        mean_abs_error_list.append(mean_abs_error)
    mean_error = np.mean(np.array(mean_abs_error_list))
    print(mean_abs_error_list, mean_error)