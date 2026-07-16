import yaml
from easydict import EasyDict
import numpy as np
import math
from sympy import Matrix
import time
import pandas as pd
import matplotlib.pyplot as plt
import random


def convert_lists_to_np_arrays(params):
    for key, value in params.items():
        if isinstance(value, list):
            params[key] = np.array(value)
        elif isinstance(value, dict) or isinstance(value, EasyDict):
            convert_lists_to_np_arrays(value)  # 递归调用
    return params


def sign(x):
    if x > 0:
        res = 1
    if x < 0:
        res = -1
    if x == 0:
        res = 0
    return res


def cal_solar(day, It, Itd, SF):
    if (day < 80):
        for i in range (0, 5):
            for j in range (0, 1440):
                Itd[j][i] = It[j][i] - SF[i] * (1/3) * ((80 - day) / 89)
                if Itd[j][i] < 0:
                    Itd[j][i] *= 0.6
    if ((day >= 80) and (day < 173)):
        for i in range (0, 5):
            for j in range (0, 1440):
                Itd[j][i] = It[j][i] + SF[i] * (1/3) * ((day - 80) / 93)
                if Itd[j][i] < 0:
                    Itd[j][i] *= 0.4
    if ((day >= 173) and (day < 266)):
        for i in range(0, 5):
            for j in range(0, 1440):
                Itd[j][i] = It[j][i] + SF[i] * (1 / 3) * ((266 - day) / 93)
                if Itd[j][i] < 0:
                    Itd[j][i] *= 0.2
    if ((day >= 266) and (day < 356)):
        for i in range(0, 5):
            for j in range(0, 1440):
                Itd[j][i] = It[j][i] - SF[i] * (1 / 3) * ((day - 266) / 90)
                if Itd[j][i] < 0:
                    Itd[j][i] *= 0.4
    if (day >= 356):
        for i in range (0, 5):
            for j in range (0, 1440):
                Itd[j][i] = It[j][i] - SF[i] * (1/3) * ((356 + 89 - day) / 89)
                if Itd[j][i] < 0:
                    Itd[j][i] *= 0.6

def cal_density(T):
    rou = 1.293 * (273.15 / (T + 273.15))
    return rou


def cal_valve_Sv(valve_position):
    s = 13100 * math.exp(-0.1801 * valve_position) + 6680 * math.exp(-0.04819 * valve_position)
    return s

def cal_pump_valve(valve_position):
    if valve_position == 0:
        s = 1000000
    else:
        s = 1000
    return s


def max_colume_irrelevant_group(A, nodes, branches):           # Return colume numbers of the max colume irrelevant group
    mat = Matrix(A)
    mat_new = mat.rref()
    A_rref = np.zeros((nodes, branches))
    for i in range(0, nodes):
        for j in range(0, branches):
            A_rref[i][j] = mat_new[0][i * branches + j]
    rows = len(A)
    #columes = len(A[0])
    flag = np.ones(rows) * -1
    for i in range(0, rows):
        for j in range(0, branches):
            if A_rref[i][j] == 1.0:
                flag[i] = j
                break
    return flag

def R_room_occ(min, room):
    if room.history_flag:
        if room.use_honeycomb:
            room.occupant_num = room.occupant_num_data[min+540]
        else:
            room.occupant_num = {"sitting": room.sitting_list[min-1],
                                 "walking": room.walking_list[min-1],
                                 "standing": room.standing_list[min-1]}
    else:
        room.occupant_num = random.randint(0, 8)

def show_origin(target, new_columes):
    length = len(new_columes)
    result = np.zeros(length)
    for i in range(0, length):
        result[int(new_columes[i])] = round(target[i][0], 1)
    return result

def clear_db(data_recorder):
    data_recorder.clear()

def init_data_recorder(data_recorder):
    # 初始化每个房间的控制参数
    for r in range(1, 8):
        room_control_key = f"room{r}_control"
        data_recorder[room_control_key] = {
            "roomtemp_setpoint": [],
            "roomRH_setpoint": [],
            "FCU_onoff_setpoint": [],
            "FCU_fan_setpoint": [],
            "FCU_workingmode_setpoint": [],
            "valve_setpoint": [],
        }

        room_key = f"room{r}"
        data_recorder[room_key] = {
            "room_temp": [],
            "room_RH": [],
            "room_Qload": [],
            "room_Qa": [],
            "outdoor_temp": [],
            "FCU_onoff_feedback": [],
            "FCU_fan_feedback": [],
            "FCU_workingmode_feedback": [],
            "supply_temp": [], #FCU 供水温度
            "return_temp": [], #FCU 回水温度
            "supply_pressure": [], #FCU 供水压力
            "return_pressure": [], #FCU 回水压力
            "waterflow": [], # FCU 流量
            "valve_feedback": [], # FCU 水阀开度【现有控制策略未使用】
            "FCU_power": [], # FCU 功耗
            "occupant_num": [], # 人数
            "occupant_transform": [],# 人员转移
        }


    # 初始化泵的控制参数
    data_recorder["pump1_control"] = {
        "pump_onoff_setpoint": [],
        "pump_frequency_setpoint": [],
        "pump_valve_setpoint": [],
    }

    data_recorder["pump1"] = {
        "pump_onoff_feedback": [], # 启停
        "pump_supplypressure": [], #
        "pump_returnpressure": [],
        "pump_flow": [],
        "pump_frequency_feedback": [],
        "pump_supplytemp": [],
        "pump_returntemp": [],
        "pump_valve_feedback": [],
    }

    # 初始化热泵的控制参数
    data_recorder["heatpump_control"] = {
        "heatpump_onoff_setpoint": [],
        "heatpump_supplytemp_setpoint": [],
        "heatpump_workingmode_setpoint": [],
    }

    data_recorder["heatpump"] = {
        "heatpump_onoff_feedback": [],
        "heatpump_supplytemp_feedback": [],
        "heatpump_workingmode_feedback": [],
        "heatpump_alarm":[],
    }

    data_recorder["sensor_outdoor"] = {
        "outdoor_temp": [],
        "outdoor_damp": [],
    }

    data_recorder["pipe_net"] = {
        "node_pressure": [],
        "branch_flow": [],
    }

    data_recorder["training"] = {
        "reward": [],
        "temperature_bias": [],
        "energy_consumption": [],
        "mean_pmv": [],
        "mean_ppd": [],
    }

def get_pipenet(G_rec, P, branches, nodes):
    G_list = [round(G_rec[g][0], 4) for g in range(branches)]
    P_list = [round(P[n][0], 1) for n in range(nodes)]

    data = {"node_pressure": P_list, "branch_flow":G_list}

    return data


def save_data(data_recorder, table_name, data):
    # 检查表名是否存在
    if table_name not in data_recorder:
        raise ValueError(f"Table '{table_name}' does not exist in data_recorder.")

    # 获取目标表
    target_table = data_recorder[table_name]

    # 检查数据是否完整
    for key in data:
        if key not in target_table:
            raise ValueError(f"Key '{key}' does not exist in table '{table_name}'.")

    # 保存数据
    for key in data:
        target_table[key].append(data[key])

# def get_latest_observation(data_recorder, key_list):
#     latest_data = []
#
#     for key in key_list:
#         # Locate the data list for the given key across all nested dictionaries
#         for sub_dict in data_recorder.values():
#             if key in sub_dict:
#                 if sub_dict[key]:  # Check if there is data in the list
#                     latest_data.append(sub_dict[key][-1])
#                 else:
#                     latest_data.append(None)  # Append None if the list is empty
#
#     return np.array(latest_data)


def get_latest_observation(data_recorder, key_dict):
    latest_data = []
    for main_key, sub_keys in key_dict.items():
        for sub_key in sub_keys:
            # 直接访问嵌套的 key
            if main_key in data_recorder and sub_key in data_recorder[main_key]:
                data = data_recorder[main_key][sub_key][-1]
                if isinstance(data, dict):
                    for k, v in data.items():
                        latest_data.append(v)
                else:
                    latest_data.append(data)
            else:
                raise KeyError(f"Key path '{main_key}-{sub_key}' does not exist in data_recorder.")
    # 转换成 numpy array
    return np.array(latest_data)


def get_latest_observation_from_every_room(data_recorder, data_name):
    room_state_keys = [data_name]
    room_state_key_dict = {"room1": room_state_keys,
                          "room2": room_state_keys,
                          "room3": room_state_keys,
                          "room4": room_state_keys,
                          "room5": room_state_keys,
                          "room6": room_state_keys,
                          "room7": room_state_keys,
                          }
    room_state_vec = get_latest_observation(data_recorder, room_state_key_dict)
    return room_state_vec


def visualize_time_point(data_recorder, time_index):
    # Prepare a dictionary to hold the data for the specified time point
    data_at_time = {}

    # Extract room data
    for r in range(1, 8):
        room_key = f"room{r}"
        room_data = {f"{room_key}_{attr}": data_recorder[room_key][attr][time_index]
                     for attr in data_recorder[room_key]}
        data_at_time.update(room_data)

        control_key = f"{room_key}_control"
        control_data = {f"{control_key}_{attr}": data_recorder[control_key][attr][time_index]
                        for attr in data_recorder[control_key]}
        data_at_time.update(control_data)

    # Extract other system data
    for key in ["pump1_control", "pump1", "heatpump_control", "heatpump", "sensor_outdoor", "pipe_net"]:
        system_data = {f"{key}_{attr}": data_recorder[key][attr][time_index]
                       for attr in data_recorder[key]}
        data_at_time.update(system_data)

    # Print each key-value pair
    for key, value in data_at_time.items():
        print(f"{key}: {value}")

# Example usage:
# visualize_time_point(data_recorder, i)




