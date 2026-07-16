import yaml
from easydict import EasyDict
import numpy as np
from common.utils import convert_lists_to_np_arrays

# def convert_lists_to_np_arrays(params):
#     for key, value in params.items():
#         if isinstance(value, list):
#             params[key] = np.array(value)
#         elif isinstance(value, dict) or isinstance(value, EasyDict):
#             convert_lists_to_np_arrays(value)  # 递归调用
#     return params

# 读取YAML文件并加载为EasyDict
with open('hyperparams/spbs_default.yml', 'r') as file:
    params = EasyDict(yaml.safe_load(file))

# 转换所有列表为 NumPy 数组
params = convert_lists_to_np_arrays(params)

# 现在可以用点操作符访问参数
PI = params.reference.PI

print(PI)
