from collections import namedtuple
import numpy as np
import torch
from .normalization import DatasetNormalizer
from .preprocessing import get_preprocess_fn
from .buffer import ReplayBuffer

# train:
Batch = namedtuple('Batch', 'trajectories conditions')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')



class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env=None, horizon=64, dataset_ratio=None,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=300,
        max_n_episodes=50000, termination_penalty=0, use_padding=True, seed=None,
        history_length=None):
        self.env = env
        ## if use history
        # change for V1 & V2
        if self.env in ["MMM2", "6h_vs_8z", "3s5z_vs_3s6z"]:
            self.use_history_tra = True
            self.use_history_length = 3
            self.use_all_concat = False
            self.use_all_with_order = False
        else:
            self.use_history_tra = False
            self.use_history_length = 3
            self.use_all_concat = False
            self.use_all_with_order = True
        print('='*10)
        print('history length: ', self.use_history_length)
        print('='*10)
        # print('max trajectory length: ', self.use_history_tra)
        ##
        self.dataset_ratio = dataset_ratio
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.horizon = horizon
        max_path_length = 500
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty
        self.use_padding = use_padding

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        
        data = self.generate_smac_dataset()
        fields = self.added_smac_data(data, fields)
        print('path length: ', fields['path_lengths'])
        print('normalizer: ', normalizer)
        self.normalizer = DatasetNormalizer(fields, normalizer, 
                            path_lengths=fields['path_lengths'])
        
        self.indices = self.make_indices(fields.path_lengths, horizon)
        
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        print('obs, act: ', self.observation_dim, self.action_dim)
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()
        print('fields: ', fields)

        # online buffer
        self.fields_online = ReplayBuffer(max_n_episodes, self.max_path_length, self.termination_penalty)
        # self.fields_online = ReplayBuffer(100, self.max_path_length, self.termination_penalty)
        self.indices_online = None
        self.online_train = False
        
    def add_online_data(self, data):
        # print(data['state'].shape, data['obs'].shape, data.keys(), '1'*10)
        # data['state']: (64, 50, 5, 114)
        self.online_train = True
        data = self.pre_online_data(data)
        data = self.normalize_online(data)
        self.fields_online = self.added_online_smac_data(data, self.fields_online)
        self.indices_online = self.make_indices(self.fields_online.path_lengths, self.horizon)
        print('online buffer: ', self.fields_online)

    def generate_smac_dataset(self):        
        # import pickle
        # print('loading')
        # file_path = '/home/yangxu/observation/sparse_data.pkl'
        # with open(file_path, 'rb') as file:
        #     data = pickle.load(file)
        # print('load done')
        
        if self.env == '3s5z_vs_3s6z' or self.env == '6h_vs_8z' or self.env == 'MMM2':
            # obs_file_path = f'/home/yyq-shixi3/observation/on-policy-diffusion/onpolicy/scripts/train_smac_scripts/data/{self.env}-obs.npy'
            # state_file_path = f'/home/yyq-shixi3/observation/on-policy-diffusion/onpolicy/scripts/train_smac_scripts/data/{self.env}-state.npy'
            obs_file_path = f'/home/yyq-shixi3/smac_data/{self.env}-obs.npy'
            state_file_path = f'/home/yyq-shixi3/smac_data/{self.env}-state.npy'
        else:
            # obs_file_path = f'/home/yyq-shixi3/observation/on-policy-diffusion/onpolicy/scripts/train_smacv2_scripts/data/{self.env}-obs.npy'
            # state_file_path = f'/home/yyq-shixi3/observation/on-policy-diffusion/onpolicy/scripts/train_smacv2_scripts/data/{self.env}-state.npy'
            # obs_file_path = f'/home/yyq-shixi3/observation/data/{self.env}-obs.npy'
            # state_file_path = f'/home/yyq-shixi3/observation/data/{self.env}-state.npy'
            obs_file_path = f'/home/yyq-shixi3/smac_data/{self.env}-obs.npy'
            state_file_path = f'/home/yyq-shixi3/smac_data/{self.env}-state.npy'

        load_obs = np.load(obs_file_path)
        load_state = np.load(state_file_path)
        concat_load_obs, concat_load_state = [], []
        
        for i in range(load_obs.shape[2]):
            if i == 0:
                concat_load_obs = load_obs[:, :, i]
                concat_load_state = load_state[:, :, i]
            else:
                concat_load_obs = np.concatenate((concat_load_obs, load_obs[:, :, i]))
                concat_load_state = np.concatenate((concat_load_state, load_state[:, :, i]))
        data = {'obs': concat_load_obs,
                'state': concat_load_state}
        print('\n', 'data load done...', '\n')
        print('obs: ', concat_load_obs.shape, 
                ' state: ', concat_load_state.shape)
        return data
    
    def pre_online_data(self, data):
        # 将数据针对每个Agent进行处理
        # 'state', 'obs', 'done', 'active_mask'
        pre_online_state, pre_online_obs, pre_online_done, pre_online_active_done = [], [], [], []
        for ii in range(len(data['state'])):
            np_array_state = np.array(data['state'][ii])
            for agent_i in range(np_array_state.shape[1]):
                added_state = np_array_state[:, agent_i] # [:, :20]
                
                added_obs = np.array(data['obs'][ii])
                if self.use_history_tra:
                    # concat_obs = self.concat_history_tra(added_obs[:, agent_i])
                    concat_obs = added_obs[:, agent_i]
                    # 拼接所有智能体的obs
                elif self.use_all_concat:
                    # concat_obs = added_obs.reshape(added_obs.shape[0], -1)
                    concat_obs = added_obs[:, agent_i]
                    # for other_agent in range(added_obs.shape[1]):
                    #     if other_agent != agent_i:
                    #         concat_obs = np.concatenate((concat_obs, added_obs[:, other_agent]), axis=1)
                elif self.use_all_with_order:
                    concat_obs = added_obs[:, agent_i]
                    # own_obs = added_obs[:, agent_i]
                    # concat_obs = added_obs.reshape(added_obs.shape[0], -1)
                    # concat_obs = np.concatenate((own_obs, concat_obs), axis=1)
                # print(concat_obs.shape,"******")
                pre_online_state.append(added_state)
                pre_online_obs.append(concat_obs)

        pre_online_data = {'actions': np.array(pre_online_state),
                           'observations': np.array(pre_online_obs),}
        return pre_online_data

    def concat_history_tra(self, added_obs):
        expanded_concat_obs = np.pad(added_obs, ((self.use_history_length - 1, 0), (0, 0)),
                                     mode='constant', constant_values=0)
        expanded_concat_obs[0:2] = added_obs[0]
        convert_obs = []
        for index in range(expanded_concat_obs.shape[0] - self.use_history_length + 1):
            concat = np.concatenate(expanded_concat_obs[index:index + self.use_history_length],
                                    axis=0)  # 在第一维拼接
            convert_obs.append(concat)
        concat_obs = np.array(convert_obs)
        return concat_obs

    def added_smac_data(self, data, fields):
        # 添加离线Offline data
        # 'state', 'obs', 'done', 'active_mask'
        for ii in range(len(data['state'])):
            np_array_state = np.array(data['state'][ii])
            for agent_i in range(np_array_state.shape[1]):
                added_state = np_array_state[:, agent_i] # [:, :20]
                
                added_obs = np.array(data['obs'][ii])
                ## 使用单个智能体历史数据
                # breakpoint()
                if self.use_history_tra:
                    concat_obs = self.concat_history_tra(added_obs[:, agent_i])
                # 拼接所有智能体的obs
                elif self.use_all_concat:
                    # concat_obs = added_obs.reshape(added_obs.shape[0], -1)
                    concat_obs = added_obs[:, agent_i]
                    for other_agent in range(added_obs.shape[1]):
                        if other_agent != agent_i:
                            concat_obs = np.concatenate((concat_obs, added_obs[:, other_agent]), axis=1)
                elif self.use_all_with_order:
                    own_obs = added_obs[:, agent_i]
                    concat_obs = added_obs.reshape(added_obs.shape[0], -1)
                    concat_obs = np.concatenate((own_obs, concat_obs), axis=1)
                    # breakpoint()
                added_episode = {'actions': added_state,
                                'observations': concat_obs,}
                fields.add_path(added_episode)
                # print(concat_obs.shape,"****")
        # print("new obs shape", concat_obs.shape)
        print('add done')
        fields.finalize()
        print('final done')
        return fields

    def added_online_smac_data(self, data, fields):
        for ii in range(len(data['observations'])):
            added_episode = {'actions': data['actions'][ii],
                            'observations': data['observations'][ii],
                            'normed_observations': data['normed_observations'][ii],
                            'normed_actions': data['normed_actions'][ii]}
            fields.add_path(added_episode)
        fields.finalize_online()
        return fields
        
    def normalize(self, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
            if key == 'actions':
                print('norm: ', key, normed.shape, normed[0])

    def normalize_online(self, data, keys=['observations', 'actions']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = data[key].reshape(data[key].shape[0]*data[key].shape[1], -1)
            print('online norm: ', key, array.shape, data[key].shape)
            normed = self.normalizer(array, key)
            normed_data = normed.reshape(data[key].shape[0], data[key].shape[1], -1)
            print('online norm: ', normed_data.shape)
            data[f'normed_{key}'] = normed_data
        return data

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = int(min(path_length - 1, self.max_path_length - horizon))
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)
    
    # online mappo:
    def __getitem__(self, idx, eps=1e-4):
        if self.online_train == False:
            path_ind, start, end = self.indices[idx]
            observations = self.fields.normed_observations[path_ind, start:end]
            actions = self.fields.normed_actions[path_ind, start:end]
            conditions = self.get_conditions(observations)
            trajectories = np.concatenate([actions, observations], axis=-1)
            batch = Batch(trajectories, conditions)
        else:
            idx_online = np.random.randint(0, len(self.indices_online)-1) 
            path_ind_online, start_online, end_online = self.indices_online[idx_online]
            observations_online = self.fields_online.normed_observations[path_ind_online, start_online:end_online]
            actions_online = self.fields_online.normed_actions[path_ind_online, start_online:end_online]
            conditions_online = self.get_conditions(observations_online)
            trajectories_online = np.concatenate([actions_online, observations_online], axis=-1)
            batch = Batch(trajectories_online, conditions_online)
        return batch