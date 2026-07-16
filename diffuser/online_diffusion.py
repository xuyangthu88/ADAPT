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

# TODO: as parameters
observation_dim = 240 # obs*n
action_dim = 114 # state


#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'hopper-medium-expert-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('diffusion')




#-----------------------------------------------------------------------------#
#------------------------------ model & trainer ------------------------------#
#-----------------------------------------------------------------------------#