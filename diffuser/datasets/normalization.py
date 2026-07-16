import numpy as np
import scipy.interpolate as interpolate
import pdb

POINTMASS_KEYS = ['observations', 'actions', 'next_observations', 'deltas']

#-----------------------------------------------------------------------------#
#--------------------------- multi-field normalizer --------------------------#
#-----------------------------------------------------------------------------#

class DatasetNormalizer:

    def __init__(self, dataset, normalizer, path_lengths=None):
        dataset = flatten(dataset, path_lengths)

        self.observation_dim = dataset['observations'].shape[1]
        self.action_dim = dataset['actions'].shape[1]
        
        if type(normalizer) == str:
            normalizer = eval(normalizer)
        
        self.means = {}
        self.stds = {}
        self.normalizers = {}
        for key, val in dataset.items():
            try:
                # save mean and std
                # change
                # val = torch.tensor(val, dtype=torch.float32)
                self.normalizers[key] = normalizer(val)
                self.means[key], self.stds[key] = self.normalizers[key].return_sta()
            except:
                print(f'[ utils/normalization ] Skipping {key} | {normalizer}')

    def __repr__(self):
        string = ''
        for key, normalizer in self.normalizers.items():
            string += f'{key}: {normalizer}]\n'
        return string

    def __call__(self, *args, **kwargs):
        return self.normalize(*args, **kwargs)
    
    def normalize(self, x, key):
        return self.normalizers[key].normalize(x)

    def unnormalize(self, x, key):
        return self.normalizers[key].unnormalize(x)

    def get_field_normalizers(self):
        return self.normalizers

def flatten(dataset, path_lengths):
    '''
        flattens dataset of { key: [ n_episodes x max_path_lenth x dim ] }
            to { key : [ (n_episodes * sum(path_lengths)) x dim ]}
    '''
    flattened = {}
    for key, xs in dataset.items():
        assert len(xs) == len(path_lengths)
        flattened[key] = np.concatenate([
            x[:int(length)]
            for x, length in zip(xs, path_lengths)
        ], axis=0)
    return flattened


#-----------------------------------------------------------------------------#
#-------------------------- single-field normalizers -------------------------#
#-----------------------------------------------------------------------------#

class Normalizer:
    '''
        parent class, subclass by defining the `normalize` and `unnormalize` methods
    '''

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins = X.min(axis=0)
        self.maxs = X.max(axis=0)

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    -: '''
            f'''{np.round(self.mins, 2)}\n    +: {np.round(self.maxs, 2)}\n'''
        )

    def __call__(self, x):
        return self.normalize(x)

    def normalize(self, *args, **kwargs):
        raise NotImplementedError()

    def unnormalize(self, *args, **kwargs):
        raise NotImplementedError()


class GaussianNormalizer(Normalizer):
    '''
        normalizes to zero mean and unit variance
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.means = self.X.mean(axis=0)
        # print(self.means.shape,"*****")
        self.stds = self.X.std(axis=0)
        # print(self.stds.shape, "^^^^^^^^")
        # breakpoint()
        self.z = 1

    def __repr__(self):
        return (
            f'''[ Normalizer ] dim: {self.mins.size}\n    '''
            f'''means: {np.round(self.means, 2)}\n    '''
            f'''stds: {np.round(self.z * self.stds, 2)}\n'''
        )

    def normalize(self, x):
        return (x - self.means) / (self.stds+0.0001)

    def unnormalize(self, x):
        return x * self.stds + self.means

    def return_sta(self):
        return self.means, self.stds


def atleast_2d(x):
    if x.ndim < 2:
        x = x[:,None]
    return x

# import torch
#
# import torch
#
# class Normalizer:
#     '''
#     Parent class, subclass by defining the `normalize` and `unnormalize` methods
#     '''
#
#     def __init__(self, X):
#         self.X = X.to(torch.float32)
#         self.mins, _ = torch.min(X, dim=0)
#         self.maxs, _ = torch.max(X, dim=0)
#
#     def __repr__(self):
#         return (
#             f'[ Normalizer ] dim: {self.mins.size(0)}\n    -: '
#             f'{torch.round(self.mins, decimals=2)}\n    +: {torch.round(self.maxs, decimals=2)}\n'
#         )
#
#     def __call__(self, x):
#         return self.normalize(x)
#
#     def normalize(self, *args, **kwargs):
#         raise NotImplementedError()
#
#     def unnormalize(self, *args, **kwargs):
#         raise NotImplementedError()
#
#
# class GaussianNormalizer(Normalizer):
#     '''
#     Normalizes to zero mean and unit variance
#     '''
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.means = torch.mean(self.X, dim=0)
#         self.stds = torch.std(self.X, dim=0)
#         self.z = 1
#
#     def __repr__(self):
#         return (
#             f'[ Normalizer ] dim: {self.mins.size(0)}\n    '
#             f'means: {torch.round(self.means, decimals=2)}\n    '
#             f'stds: {torch.round(self.z * self.stds, decimals=2)}\n'
#         )
#
#     def normalize(self, x):
#         return (x - self.means) / (self.stds + 0.0001)
#
#     def unnormalize(self, x):
#         return x * self.stds + self.means
#
#     def return_sta(self):
#         return self.means, self.stds
#
#
# def atleast_2d(x):
#     if x.dim() < 2:
#         x = x.unsqueeze(1)
#     return x
#


