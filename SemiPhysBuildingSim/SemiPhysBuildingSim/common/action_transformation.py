from gym.spaces import Dict, Discrete, Box
import gym
import numpy as np
# Action is Discrete Now

def available_action_set():
    action = {}

    # 初始化每个房间的控制参数
    for r in range(1, 8):
        room_control_key = f"room{r}_control"
        action[room_control_key] = {
            "roomtemp_setpoint": {24}, #
            "roomRH_setpoint": {70},# 固定70
            "FCU_onoff_setpoint": {1}, # 选{0，1}最简单控制
            "FCU_fan_setpoint": {0,1,2,3}, #{0,3}, #{0,1,2,3}, # 选{1，2，3}低中高档 {0,1,3},#
            "FCU_workingmode_setpoint": {1},
            "valve_setpoint": {100},# 固定100
        }

    print("FCU_fan_setpoint: " + str(action["room1_control"]["FCU_fan_setpoint"]))
    # 初始化泵的控制参数[暂时固定]
    action["pump1_control"] = {
        "pump_onoff_setpoint": {1},
        "pump_frequency_setpoint": {50},
        "pump_valve_setpoint": {1},
    }

    # 初始化热泵的控制参数【暂时固定】
    action["heatpump_control"] = {
        "heatpump_onoff_setpoint": {1},
        "heatpump_supplytemp_setpoint": {7},
        "heatpump_workingmode_setpoint": {2},
    }
    return action

def create_action_space(action_set):
    action_space = []
    for room in action_set.keys():
        params = action_set[room]
        # 获取每个控制参数的取值个数
        sizes = [len(v) for v in params.values()]
        action_space.append(sizes)

    # 计算总的离散动作组合数量
    total_actions = np.prod([np.prod(size) for size in action_space])
    return gym.spaces.Discrete(total_actions)


def map_action_to_controls(action_set, action_index):
    controls = {}
    # 从action_set中获取控制参数
    for room, params in action_set.items():
        controls[room] = {}
        param_indices = {key: 0 for key in params.keys()}

        # 根据索引计算出每个控制参数的取值
        for key, value_set in params.items():
            value_list = list(value_set)
            size = len(value_list)
            param_index = (action_index % size)
            param_indices[key] = value_list[param_index]
            action_index //= size

        controls[room] = param_indices

    return controls


def map_controls_to_action(action_set, controls):
    action_index = 0
    multiplier = 1
    for room, params in action_set.items():
        room_control = controls[room]

        for key, value_set in params.items():
            value_list = list(value_set)
            value_index = value_list.index(room_control[key])  # 找到控制参数的索引
            action_index += value_index * multiplier
            multiplier *= len(value_list)  # 更新乘数以支持多维组合

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