from gym.spaces import Dict, Discrete, Box
import gym
import numpy as np
from .action_transformation import available_action_set
# Action is Multi Discrete Now

# def available_action_set():
#     action = {}
#
#     # 初始化每个房间的控制参数
#     for r in range(1, 8):
#         room_control_key = f"room{r}_control"
#         action[room_control_key] = {
#             "roomtemp_setpoint": {24}, #
#             "roomRH_setpoint": {70},# 固定70
#             "FCU_onoff_setpoint": {1}, # 选{0，1}最简单控制
#             "FCU_fan_setpoint": {0,3}, # 选{1，2，3}低中高档
#             "FCU_workingmode_setpoint": {1},
#             "valve_setpoint": {100},# 固定100
#         }
#
#     # 初始化泵的控制参数[暂时固定]
#     action["pump1_control"] = {
#         "pump_onoff_setpoint": {1},
#         "pump_frequency_setpoint": {50},
#         "pump_valve_setpoint": {1},
#     }
#
#     # 初始化热泵的控制参数【暂时固定】
#     action["heatpump_control"] = {
#         "heatpump_onoff_setpoint": {1},
#         "heatpump_supplytemp_setpoint": {7},
#         "heatpump_workingmode_setpoint": {2},
#     }
#     return action

def create_action_space(action_set):
    action_space = []
    for room in action_set.keys():
        params = action_set[room]
        # 获取每个控制参数的取值个数
        for v in params.values():
            sizes = len(v)
            if sizes == 1:
                pass
            else:
                action_space.append(sizes)

    return gym.spaces.MultiDiscrete(action_space)


def map_action_to_controls(action_set, action_index):
    controls = {}
    total_control_lenth = len(action_index)
    current_control_index = 0

    # 从action_set中获取控制参数
    for room, params in action_set.items():
        controls[room] = {}
        param_indices = {key: 0 for key in params.keys()}

        # 根据索引计算出每个控制参数的取值
        for key, value_set in params.items():
            value_list = list(value_set)
            size = len(value_list)
            if size == 1:
                param_indices[key] = value_list[0]
            else:
                current_control = action_index[current_control_index]
                param_indices[key] = value_list[current_control]
                current_control_index += 1

        controls[room] = param_indices

    assert total_control_lenth == current_control_index,\
        "Error: total control length not equal to current control index"

    return controls


def map_controls_to_action(action_set, controls):
    action_index = []
    for room, params in action_set.items():
        room_control = controls[room]

        for key, value_set in params.items():
            value_list = list(value_set)
            size = len(value_list)
            if size == 1:
                pass
            else:
                value_index = value_list.index(room_control[key]) # 找到当前控制参数的索引
                action_index.append(value_index)
    action_index = np.array(action_index)
    return action_index


if __name__ == "__main__":
    # 使用示例
    action_set = available_action_set()
    action_space = create_action_space(action_set)

    # 从动作空间中采样
    sampled_action_index = action_space.sample()
    print(f"Sampled Action Index: {sampled_action_index}")

    controls = map_action_to_controls(action_set, sampled_action_index)
    # 将控制指令映射到动作索引
    mapped_action_index = map_controls_to_action(action_set, controls)
    print(f"Mapped Action Index: {mapped_action_index}")


    print("Mapped Controls:", controls)