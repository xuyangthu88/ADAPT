import diffuser.sampling as sampling
import diffuser.utils as utils
import algos

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    dataset: str = "10gen_zerg5v5"
    # MMM2
    # 3s5z_vs_3s6z
    # 6h_vs_8z
    # 10gen_terran10v11 10gen_terran20v20
    # 10gen_zerg10v10 10gen_zerg10v11
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

#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

import numpy as np
import os

## load diffusion model and value function from disk
load_path = 'MA_H_' + str(args.horizon) + '_Dim_' + str(args.dim_mults) + '_T_' + str(args.n_diffusion_steps) + '_ratio_' + str(args.data_ratio) + '_' + str(args.dataset) + '_' + str(args.loss_type) + '_new'

diffusion_experiment = utils.load_diffusion(
    args.loadbase, load_path, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)

diffusion = diffusion_experiment.ema
dataset = diffusion_experiment.dataset

## policies are wrappers around an unconditional diffusion model and a value guide
policy_config = utils.Config(
    args.policy,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)
policy = policy_config()

#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#
# change for z
def get_pretrained_p_net():
    return diffusion_experiment.model_p, diffusion_experiment.model_p_ema

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
    return state.reshape(bs,n_agents,-1) # [bs,n,dim]

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

def update_diffusion_policy(data):
    print('update diffusion...')
    trainer_config = utils.Config(
        utils.Trainer,
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

    output_log = trainer.train(n_train_steps=10000)
    return output_log

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