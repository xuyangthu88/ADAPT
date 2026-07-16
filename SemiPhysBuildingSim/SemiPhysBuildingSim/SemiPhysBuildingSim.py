import time
import numpy as np
import matplotlib.pyplot as plt
import gym
import math
from easydict import EasyDict
from .common.utils import convert_lists_to_np_arrays, sign, max_colume_irrelevant_group, init_data_recorder, \
    get_pipenet, save_data, cal_solar, get_latest_observation, cal_pump_valve, cal_valve_Sv,show_origin, \
    cal_density, R_room_occ, get_latest_observation_from_every_room
import yaml
import csv
from pathlib import Path
from .common.mz_model import ZONE, FCU, PUMP, HEATPUMP
from sympy import Matrix
import copy
# False # True #
# USE_Multi_Discrete = False #
# if USE_Multi_Discrete:
#     from .common.action_transformation_multi_discrete import available_action_set, create_action_space, \
#         map_action_to_controls, map_controls_to_action
# else:
#     from .common.action_transformation import available_action_set, create_action_space, \
#         map_action_to_controls, map_controls_to_action

from .common.pmvppd_lookup import SimplifiedPMVPPDLookup

class SemiPhysBuildingSimulation(gym.core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 hyperparams=None,
                 hyperparams_path='SemiPhysBuildingSim/hyperparams/spbs_default.yml',
                 reward_mode = "Baseline_OCC_PPD_without_energy",
                 tradeoff_constant = 100,
                 eval_mode=False,
                 USE_Multi_Discrete = False):

        # USE_Multi_Discrete = False  #
        self.USE_Multi_Discrete = USE_Multi_Discrete
        if USE_Multi_Discrete:
            from .common.action_transformation_multi_discrete import available_action_set, create_action_space, \
                map_action_to_controls, map_controls_to_action
            self.map_action_to_controls = map_action_to_controls
            self.map_controls_to_action = map_controls_to_action
        else:
            from .common.action_transformation import available_action_set, create_action_space, \
                map_action_to_controls, map_controls_to_action
            self.map_action_to_controls = map_action_to_controls
            self.map_controls_to_action = map_controls_to_action

        # Read from hyperparams or hyperparams_path
        if hyperparams is None:
            with open(hyperparams_path, 'r') as file:
                default_hyperparams = EasyDict(yaml.safe_load(file))
            hyperparams = convert_lists_to_np_arrays(params=default_hyperparams)
        self.hyperparams = hyperparams

        self.eval_mode = eval_mode

        # Set the reward mode & trardoff constant for energy consumption
        self.reward_mode_list = ["Baseline_without_energy",
                                 "Baseline_with_energy",
                                 "Baseline_OCC_PPD_without_energy",
                                 "Baseline_OCC_PPD_with_energy",
                                 "Baseline_occ_with_energy_new",
                                 "Baseline_occ_with_energy_sm"
                                 ]
        self.reward_mode = reward_mode
        self.tradeoff_constant = tradeoff_constant  # 10 or 100
        self.smoothness_constant = getattr(self.hyperparams, "smoothness_constant", 0.0)
        # self.track_smoothness = self.reward_mode == "Baseline_occ_with_energy_sm"
        self.track_smoothness = self.reward_mode == "Baseline_OCC_PPD_with_energy"
        self.prev_action = None
        self.current_step_smoothness = 0.0
        self.smoothness_history = []

        print("Reward mode: ", self.reward_mode)
        print("Tradeoff constant: ", self.tradeoff_constant)


        # Store all the data in this Dict
        self.data_recorder = {}
        self.step_min = 0
        self.prev_action = None
        self.current_step_smoothness = 0.0
        if self.track_smoothness:
            self.smoothness_history.clear()
        self.step_min_bound = 600

        # self.action_dict_space = action_dict_space()
        self.available_action_set = available_action_set()
        self.action_space = create_action_space(action_set=self.available_action_set)
        print("Action space: ", self.action_space)

        room_state_keys = ["room_temp",  # 房间温度
                           "FCU_fan_feedback",  # FCU 挡位
                           "supply_temp",  # FCU 供水温度
                           "return_temp",  # FCU 回水温度
                           # "supply_pressure",  # FCU 供水压力
                           # "return_pressure",  # FCU 回水压力
                           "occupant_num",  # 房间人数
                           ]

        self.observation_key_dict = {"sensor_outdoor": ["outdoor_temp"],  # 室外温度
                                     "room1": room_state_keys,
                                     "room2": room_state_keys,
                                     "room3": room_state_keys,
                                     "room4": room_state_keys,
                                     "room5": room_state_keys,
                                     "room6": room_state_keys,
                                     "room7": room_state_keys,
                                     }
        obs_shape = 4*7 + 1 + 3*7

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float32)

        # # Generate framework from hyperparams
        # self.reference_initialize()
        # self.mz_model_initialize()
        # self.simulator_initialize()
        #
        # # Store all the data in this Dict
        # self.data_recorder = {}
        # init_data_recorder(self.data_recorder)

    def reset(self):
        # Generate framework from hyperparams
        self.reference_initialize()
        self.mz_model_initialize()
        self.simulator_initialize()

        # initialize the data recorder
        self.data_recorder.clear()
        init_data_recorder(self.data_recorder)

        self.step_min = 0

        # TODO 1: write the first data into data_recorder
        self.G_rec = self.G_init

        # 1st: pipe_net
        pipenet_status = get_pipenet(G_rec=self.G_rec, P=self.P, branches=self.branches, nodes=self.nodes)
        save_data(self.data_recorder, "pipe_net", pipenet_status)

        # 2nd: ith-room_control
        room_control_initial = {"roomtemp_setpoint": 24.0,
                                'roomRH_setpoint': 70.0,
                                "FCU_onoff_setpoint": 1,
                                "FCU_fan_setpoint": 1,
                                "FCU_workingmode_setpoint": 1,
                                "valve_setpoint": 100, }
        for r in range(1, 8):
            save_data(self.data_recorder, 'room' + str(r) + '_control', room_control_initial)

        # 3nd: ith-room status
        for r in range(1, 8):
            room = self.room_list[r - 1]
            fcu = self.fcu_list[r - 1]
            outdoor_temp = self.dry_bulb_his[self.step_hour]
            room_status = {
                'room_temp': room.temp,
                'room_RH': room.RH,
                'room_Qload': room.Q_load / 60,
                'room_Qa': room.Qa / 60,
                'outdoor_temp': outdoor_temp,
                'FCU_onoff_feedback': fcu.onoff,
                'FCU_fan_feedback': fcu.fan_position,
                'FCU_workingmode_feedback': fcu.mode,
                'supply_temp': fcu.tw_supply,
                'return_temp': fcu.tw_return,
                'supply_pressure': fcu.tw_supplypressure,
                'return_pressure': fcu.tw_returnpressure,
                'waterflow': fcu.waterflow,
                'valve_feedback': fcu.valve_position,
                'FCU_power': fcu.power,
                'occupant_num': room.occupant_num,
                'occupant_transform': room.occupant_trans,
            }

            save_data(self.data_recorder, 'room' + str(r), room_status)

        # 4th: pump
        pump = self.pump_list[0]
        pump_status = {
            'pump_onoff_feedback': pump.onoff,
            'pump_supplypressure': pump.p_supply,
            'pump_returnpressure': pump.p_return,
            'pump_flow': pump.G,
            'pump_frequency_feedback': pump.n,
            'pump_supplytemp': pump.supply_temp,
            'pump_returntemp': pump.return_temp,
            'pump_valve_feedback': pump.valve_position,
        }
        save_data(self.data_recorder, 'pump1', pump_status)

        # 5th: heatpump
        heatpump = self.heatpump
        heatpump_status = {
            'heatpump_onoff_feedback': heatpump.onoff,
            'heatpump_supplytemp_feedback': heatpump.supplytemp,
            'heatpump_workingmode_feedback': heatpump.mode,
            'heatpump_alarm': heatpump.flow_alarm,
        }
        save_data(self.data_recorder, "heatpump", heatpump_status)

        # 6th: outdoor temp & damp
        outdoor_status = {
            'outdoor_temp': self.dry_bulb_his[self.step_hour],
            'outdoor_damp': self.damp_his[self.step_hour],
        }
        save_data(self.data_recorder, "sensor_outdoor", outdoor_status)

        # TODO 2: Get the state from data_recorder

        state = self.get_state_from_datarecorder()
        info = {}
        return state, info

    def step(self, action):
        # action = action.copy()
        next_state, r, done, info = None, 0, False, {}

        self.step_min += 1
        done = False

        if self.step_min % 60 == 0:
            self.step_hour += 1

        # Time set
        minute = ((self.init_time % 24) * 60 + self.step_min) % (24 * 60)  # The minute of the day
        day = (self.init_time / 24 + int(self.step_min / 1440)) % 365 + 1
        if self.step_min == 1:
            self.day_rec = day
        if (self.step_min == 1) or (day != self.day_rec):
            cal_solar(day, It=self.It, Itd=self.Itd, SF=self.hyperparams.reference.SF)
            self.day_rec = day

        return_sum = 0

        # TODO 3: Given Action in form of Dict, modify it
        controls_dict = self.map_action_to_controls(action_set=self.available_action_set, action_index=action)

        # Compute smoothness based on per-room actions
        if self.track_smoothness:
            if getattr(self, "USE_Multi_Discrete", False):
                # MultiDiscrete: action already encodes per-dimension controls
                action_arr = np.asarray(action, dtype=np.float32)
            else:
                # Discrete: decompose scalar action index into per-room / per-control indices
                flat_indices = []
                for room_key, params in self.available_action_set.items():
                    room_controls = controls_dict[room_key]
                    for key, value_set in params.items():
                        value_list = list(value_set)
                        if len(value_list) <= 1:
                            continue
                        current_value = room_controls[key]
                        try:
                            idx = value_list.index(current_value)
                        except ValueError:
                            # Fallback: should not happen if action sets are consistent
                            idx = 0
                        flat_indices.append(idx)
                action_arr = np.asarray(flat_indices, dtype=np.float32)

            smoothness = 0.0
            if self.prev_action is not None:
                diff = action_arr - self.prev_action
                # smoothness = float(np.sum(np.square(diff)))
                smoothness = float(np.sum(np.abs(diff)))
            self.current_step_smoothness = smoothness
            # self.smoothness_history.append(smoothness)
            self.prev_action = action_arr.copy()
        else:
            self.current_step_smoothness = 0.0

        # TODO 4: Write Transition Function

        # Step 1: Read room[r] FCU control
        # FCU_onoff_setpoint + FCU_fan_setpoint + FCU_workingmode_setpoint + valve_setpoint
        # ----> FCU_list[r].***_set ---[clip for single step]--> FCU_list[r].***_position
        for r in range(0, 7):
            room_name = 'room' + str(r + 1) + '_control'
            self.fcu_list[r].fan_set = controls_dict[room_name]["FCU_fan_setpoint"]
            self.fcu_list[r].mode_set = controls_dict[room_name]["FCU_workingmode_setpoint"]
            self.fcu_list[r].onoff_set = controls_dict[room_name]["FCU_onoff_setpoint"]
            self.fcu_list[r].valve_set = controls_dict[room_name]["valve_setpoint"]

            self.fcu_list[r].fan_position = self.fcu_list[r].fan_set
            self.fcu_list[r].mode = self.fcu_list[r].mode_set
            self.fcu_list[r].onoff = self.fcu_list[r].onoff_set

            valve_stepmax = self.hyperparams.reference.valve_stepmax
            if abs(self.fcu_list[r].valve_set - self.fcu_list[r].valve_position) > valve_stepmax:
                self.fcu_list[r].valve_position += sign(
                    self.fcu_list[r].valve_set - self.fcu_list[r].valve_position) * valve_stepmax
            else:
                self.fcu_list[r].valve_position = self.fcu_list[r].valve_set

        # Step 2: The same for pump
        self.pump_list[0].valve_set = controls_dict["pump1_control"]["pump_valve_setpoint"]
        self.pump_list[0].onoff_set = controls_dict["pump1_control"]["pump_onoff_setpoint"]
        self.pump_list[0].n_set = controls_dict["pump1_control"]["pump_frequency_setpoint"]

        self.pump_list[0].valve_position = self.pump_list[0].valve_set
        self.pump_list[0].onoff = self.pump_list[0].onoff_set

        pump_stepmax = self.hyperparams.reference.pump_stepmax
        if abs(self.pump_list[0].n_set - self.pump_list[0].n) > pump_stepmax:
            self.pump_list[0].n += sign(self.pump_list[0].n_set - self.pump_list[0].n) * pump_stepmax
        else:
            self.pump_list[0].n = self.pump_list[0].n_set

        # Step 3: The same for heatpump: onoff + mode + supplytemp ---> HeatPump
        self.heatpump.mode = controls_dict["heatpump_control"]["heatpump_workingmode_setpoint"]
        self.heatpump.supply_tempset = controls_dict["heatpump_control"]["heatpump_supplytemp_setpoint"]
        self.heatpump.onoff = controls_dict["heatpump_control"]["heatpump_onoff_setpoint"]

        # Step 4:
        # Cal resistance
        branches = self.branches
        A = self.A
        G = self.G
        S = self.S
        # from common.utils import cal_pump_valve, cal_valve_Sv

        s_valve = np.zeros((branches, branches))
        for j in range(0, branches):
            if A[1][j] == 1 and A[20][j] == -1:
                s_valve[j][j] = cal_pump_valve(self.pump_list[0].valve_position)
        for i in range(3, 11):
            for j in range(0, branches):
                if A[i][j] == 1 and A[i + 9][j] == -1:
                    s_valve[j][j] = cal_valve_Sv(self.fcu_list[self.fcu_order[i - 3]].valve_position)
                    self.fcu_list[self.fcu_order[i - 3]].waterflow = G[j][0]
        s_total = S + s_valve
        # Cal pipe_net
        max_iter = self.hyperparams.reference.max_iter
        pump1 = self.pump_list[0]
        pump1_branch = self.pump1_branch
        DH = self.DH
        abs_G = self.abs_G
        pipe_z = self.pipe_z
        nodes = self.nodes
        delta_Ht = self.delta_Ht
        At_invT = self.At_invT
        Bf = self.Bf
        Gl = self.Gl
        dDH_dG = self.dDH_dG
        diff = self.diff

        for iter in range(0, max_iter):
            pump1.G = G[pump1_branch][0]
            # pump2.G = G[pump2_branch][0]
            pump1.cal_pump()
            # pump2.cal_pump()
            DH[pump1_branch][0] = pump1.H
            # DH[pump2_branch][0] = pump2.H
            SGG = np.dot(np.dot(s_total, abs_G), G)
            delta_H = SGG + 9800 * pipe_z - DH
            for i in range(0, nodes):
                delta_Ht[i][0] = delta_H[i][0]
            P = np.dot(At_invT, delta_Ht)
            F = np.dot(Bf, delta_H)
            if F.max() <= 1 and F.min() >= -1:
                G = np.dot(Bf.T, Gl)
                break
            else:
                dDH_dG[pump1_branch][pump1_branch] = pump1.dH_dG
                # dDH_dG[pump2_branch][pump2_branch] = pump2.dH_dG
                dF_dGl = np.dot(np.dot(Bf, (2 * np.dot(s_total, abs_G) - dDH_dG)), Bf.T)
                u = np.linalg.eigvals(dF_dGl)
                dGl = np.dot(np.linalg.inv(dF_dGl), F)
                if iter <= 10:
                    step_ratio = 0.5
                if iter > 10 and iter <= 50:
                    step_ratio = 0.1
                else:
                    step_ratio = 0.05
                for i in range(0, diff):
                    Gl[i][0] -= dGl[i][0] * step_ratio
                G = np.dot(Bf.T, Gl)

        # Step 5: # Cal room & pump
        # from common.utils import show_origin, cal_density, R_room_occ
        sup_temp = self.hyperparams.reference.sup_temp

        G_rec = show_origin(G, self.new_columes)
        P -= (P[11][0] - 10000) * np.ones((nodes, 1))


        all_G_fan = []
        # print("action: " + str(action))

        for r in range(0, 7):
            rou = cal_density(sup_temp)

            self.fcu_list[r].cal_fan_G()
            self.fcu_list[r].cal_power()
            self.fcu_list[r].cal_supplyair(self.room_list[r].temp)
            R_room_occ(self.step_min, self.room_list[r])
            all_G_fan.append(self.fcu_list[r].G_fan)

        for r in range(0, 7):
            self.room_list[r].cal_room(self.dry_bulb_his[self.step_hour], self.step_min, rou, self.fcu_list[r].G_fan,
                                       all_G_fan=all_G_fan)
            self.fcu_list[r].cal_returntemp(self.room_list[r].Qa / 60)
            return_sum += 0.95 * self.fcu_list[r].waterflow * self.fcu_list[r].tw_return
            self.fcu_list[r].tw_supplypressure = P[r + 3][0]
            self.fcu_list[r].tw_returnpressure = P[r + 12][0]

        pump1.p_return = P[1][0]
        pump1.cal_pump()
        pump1.G = G_rec[26]
        pump1.return_temp = return_sum / pump1.G

        # TODO 5: Save data in self.data_recoder
        pipenet_status = get_pipenet(G_rec=self.G_rec, P=self.P, branches=self.branches, nodes=self.nodes)
        save_data(self.data_recorder, "pipe_net", pipenet_status)

        # room record
        for r in range(0, 7):
            room = self.room_list[r]
            outdoor_temp = self.dry_bulb_his[self.step_hour]
            fcu = self.fcu_list[r]
            table_name = 'room' + str(r + 1)

            rth_room_status = {
                "room_temp": room.temp,
                "room_RH": room.RH,
                "room_Qload": room.Q_load / 60,
                "room_Qa": room.Qa / 60,
                "outdoor_temp": outdoor_temp,
                "FCU_onoff_feedback": fcu.onoff,
                "FCU_fan_feedback": fcu.fan_position,
                "FCU_workingmode_feedback": fcu.mode,
                "supply_temp": fcu.tw_supply,
                "return_temp": fcu.tw_return,
                "supply_pressure": fcu.tw_supplypressure,
                "return_pressure": fcu.tw_returnpressure,
                "waterflow": fcu.waterflow,
                "valve_feedback": fcu.valve_position,
                "FCU_power": fcu.power,
                "occupant_num": room.occupant_num,
                "occupant_transform": room.occupant_trans
            }
            save_data(self.data_recorder, table_name=table_name, data=rth_room_status)

            save_data(self.data_recorder, table_name=table_name + "_control",
                      data=controls_dict[table_name + "_control"])

        # pump record
        pump = pump1
        pump_status = {
            "pump_onoff_feedback": pump.onoff,
            "pump_supplypressure": pump.p_supply,
            "pump_returnpressure": pump.p_return,
            "pump_flow": pump.G,
            "pump_frequency_feedback": pump.n,
            "pump_supplytemp": pump.supply_temp,
            "pump_returntemp": pump.return_temp,
            "pump_valve_feedback": pump.valve_position
        }
        save_data(self.data_recorder, table_name="pump1", data=pump_status)
        save_data(self.data_recorder, table_name="pump1" + "_control", data=controls_dict["pump1" + "_control"])

        # heatpump record
        heatpump = self.heatpump
        heatpump_status = {
            "heatpump_onoff_feedback": heatpump.onoff,
            "heatpump_supplytemp_feedback": heatpump.supplytemp,
            "heatpump_workingmode_feedback": heatpump.mode,
            "heatpump_alarm": heatpump.flow_alarm
        }
        save_data(self.data_recorder, table_name="heatpump", data=heatpump_status)
        save_data(self.data_recorder, table_name="heatpump" + "_control", data=controls_dict["heatpump" + "_control"])

        outdoor_status = {
            "outdoor_temp": self.dry_bulb_his[self.step_hour],
            "outdoor_damp": self.damp_his[self.step_hour]
        }
        save_data(self.data_recorder, table_name="sensor_outdoor", data=outdoor_status)

        # TODO 6: Get State from  self.data_recoder
        next_state = self.get_state_from_datarecorder()

        # TODO 7: get reward
        r = self.get_reward_from_datarecorder()
        save_data(self.data_recorder, table_name="training", data={"reward": r})
        # if self.track_smoothness:
        #     save_data(self.data_recorder, table_name="training", data={"action_smoothness": self.current_step_smoothness})

        # TODO 8: get the evaluation of the total temperature bias, total energy consumption, mean pmv and mean ppd
        if self.eval_mode:
            temperature_bias = self.get_temperature_bias_from_datarecorder_2nd()
            energy_consumption = self.get_energy_consumption_from_datarecorder()
            mean_pmv, mean_ppd = self.get_pmv_ppd_from_datarecorder()
            save_data(self.data_recorder, table_name="training", data={"temperature_bias": temperature_bias})
            save_data(self.data_recorder, table_name="training", data={"energy_consumption": energy_consumption})
            save_data(self.data_recorder, table_name="training", data={"mean_pmv": mean_pmv})
            save_data(self.data_recorder, table_name="training", data={"mean_ppd": mean_ppd})

        if self.step_min >= self.step_min_bound:
            done = True

        return next_state, r, done, done, info

    def reference_initialize(self):
        SF = self.hyperparams.reference.SF
        PI = self.hyperparams.reference.PI
        dry_bulb_file_path = self.hyperparams.reference.dry_bulb_file_path
        damp_file_path = self.hyperparams.reference.damp_file_path
        adj_file_path = self.hyperparams.reference.adj_file_path
        res_file_path = self.hyperparams.reference.res_file_path

        self.dry_bulb_his = np.zeros(8760)
        self.damp_his = np.zeros(8760)

        self.It = np.zeros((1440, 5))
        self.Itd = np.zeros((1440, 5))
        # SF = self.hyperparams.reference.SF
        # PI = self.hyperparams.reference.PI
        for i in range(0, 5):
            for j in range(0, 1440):
                self.It[j][i] = SF[i] * math.cos((j - 720) * PI / 720)

        # Climate
        # dry_bulb_file_path = self.hyperparams.reference.dry_bulb_file_path
        # damp_file_path = self.hyperparams.reference.damp_file_path

        dry_bulb_file = open(dry_bulb_file_path)
        reader = csv.reader(dry_bulb_file)
        data = list(reader)
        for i in range(0, 8760):
            self.dry_bulb_his[i] = float(data[0][i])
        dry_bulb_file.close()

        damp_file = open(damp_file_path)
        reader = csv.reader(damp_file)
        data = list(reader)
        for i in range(0, 8760):
            self.damp_his[i] = float(data[0][i])
        damp_file.close()

        #  Adj matrix
        # adj_file_path = self.hyperparams.reference.adj_file_path
        adj_file = open(adj_file_path)
        reader = csv.reader(adj_file)
        data = list(reader)
        nodes = len(data)
        self.nodes = nodes
        branches = len(data[0])
        self.branches = branches
        diff = branches - nodes
        self.diff = diff
        A_init = np.zeros((nodes, branches))
        self.A_init = A_init
        for i in range(0, nodes):
            for j in range(0, branches):
                if (data[i][j] == '1') or (data[i][j] == '-1'):
                    A_init[i][j] = int(data[i][j])
        adj_file.close()

        #  Resistance list
        # res_file_path = self.hyperparams.reference.res_file_path
        res_file = open(res_file_path)
        reader = csv.reader(res_file)
        data = list(reader)
        pipe_d_init = np.zeros(branches)
        pipe_area_init = np.zeros(branches)
        pipe_length_init = np.zeros(branches)
        pipe_local_s_init = np.zeros(branches)
        pipe_z_init = np.zeros((branches, 1))

        self.pipe_d_init = pipe_d_init
        self.pipe_area_init = pipe_area_init
        self.pipe_length_init = pipe_length_init
        self.pipe_local_s_init = pipe_local_s_init
        self.pipe_z_init = pipe_z_init

        for i in range(0, branches):
            pipe_d_init[i] = float(data[0][i])
            pipe_area_init[i] = float(data[1][i])
            pipe_length_init[i] = float(data[2][i])
            pipe_local_s_init[i] = float(data[3][i])
            pipe_z_init[i][0] = float(data[4][i])
        res_file.close()

        # Calculation matrices
        # Init
        self.G_init = np.ones((branches, 1))  # Volume flow of nodes  m3/s
        self.Q_init = np.zeros((nodes, 1))  # Air integration of nodes (only occurs in zones)  m3/s
        self.abs_G_init = np.eye(branches)  # Absolute value
        self.delta_H_init = np.zeros((branches, 1))  # Pressure loss of branches
        self.S_init = np.eye(branches)  # Resistance of branches
        self.DH_init = np.zeros((branches, 1))  # Fan head   Pa

        # Rearranged
        self.pipe_d = np.zeros(branches)
        self.pipe_area = np.zeros(branches)
        self.pipe_length = np.zeros(branches)
        self.pipe_local_s = np.zeros(branches)
        self.pipe_z = np.zeros((branches, 1))
        self.A = np.zeros((nodes, branches))
        self.G = np.zeros((branches, 1))
        self.G_min = np.zeros((branches, 1))  # Sum G in the min
        self.Q = np.zeros((nodes, 1))
        self.abs_G = np.zeros((branches, branches))
        self.P = np.zeros((nodes, 1))  # Static pressure of nodes  Pa
        self.delta_H = np.zeros((branches, 1))
        self.delta_Ht = np.zeros((nodes, 1))
        self.S = np.zeros((branches, branches))
        self.DH = np.zeros((branches, 1))
        self.F = np.zeros((diff, 1))
        self.SGG = np.zeros((branches, 1))
        self.Gl = np.zeros((diff, 1))
        self.dGl = np.zeros((diff, 1))
        self.dDH_dG = np.zeros((branches, branches))
        self.G_rec = np.zeros((branches, 1))
        self.dF_dGl = np.zeros((diff, diff))
        self.dS_dG = np.zeros((branches, branches))
        self.ddeltaPv_dG = np.zeros((branches, branches))
        self.Cq = np.zeros(branches)

        self.At = np.zeros((nodes, nodes))
        self.Al = np.zeros((nodes, diff))

        self.zone_result = np.ones(5)
        self.duct_result = np.zeros(branches)

    def mz_model_initialize(self):
        #  Zone objects
        self.room1 = ZONE('room1', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room2 = ZONE('room2', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room3 = ZONE('room3', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room4 = ZONE('room4', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room5 = ZONE('room5', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room6 = ZONE('room6', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room7 = ZONE('room7', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)
        self.room8 = ZONE('room8', 49, 5, hyperparams=self.hyperparams, Itd=self.Itd)

        self.room_list = [self.room1, self.room2, self.room3, self.room4, self.room5, self.room6, self.room7,
                          self.room8]

        wall_area = self.hyperparams.reference.wall_area
        for i in range(0, 8):
            leakage = 0
            for j in range(0, 4):
                self.room_list[i].wall_area[j] = wall_area[i][j]
                leakage += wall_area[i][j]
            self.room_list[i].leakage_area = 0.00028 * (leakage + self.room_list[i].area)
        self.room_order = np.array([7, 5, 2, 6, 4, 3, 0, 1])

        # FCU objects
        # FP102 = FCU(1, 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907)
        # FP136 = FCU(2, 1360, 1020, 680, 11381, 9376, 6954, 7282, 6250, 4890, 134, 1325)
        self.fcu1 = FCU('fcu1', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu2 = FCU('fcu2', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu3 = FCU('fcu3', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu4 = FCU('fcu4', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu5 = FCU('fcu5', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu6 = FCU('fcu6', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu7 = FCU('fcu7', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)
        self.fcu8 = FCU('fcu8', 1020, 765, 510, 8720, 7134, 5282, 5558, 4860, 3786, 96, 907,
                        hyperparams=self.hyperparams)

        self.fcu_list = [self.fcu1, self.fcu2, self.fcu3, self.fcu4, self.fcu5, self.fcu6, self.fcu7, self.fcu8]
        self.fcu_order = np.array([7, 5, 2, 6, 4, 3, 0, 1])

        # pump objects
        self.pump1 = PUMP('pump1', hyperparams=self.hyperparams)
        self.pump2 = PUMP('pump2', hyperparams=self.hyperparams)
        self.pump_list = [self.pump1, self.pump2]

        # heatpump object
        self.heatpump = HEATPUMP()

    def simulator_initialize(self):
        # Initialization
        # Time
        # self.init_time = 8025
        self.init_time = 5289 #5289  #5289 = August 9th 9:00am  # 索引 2673 = 4月22日 9:00am #11月7日 9:00am = 第 7449 小时 = 索引 7449
        self.step_hour = self.init_time
        self.T_delta = 6  # Real time interval for 1 min simulation
        self.T_con_fcu = 5  # Time interval for fcu control in simulation, min
        self.T_con_pump = 15  # Time interval for pump control in simulation, min
        self.T_con_hp = 10  # Time interval for hp control in simulation, min

        # Initialize matrices
        branches = self.branches
        G_init = self.G_init
        pipe_area_init = self.pipe_area_init
        abs_G_init = self.abs_G_init
        S_init = self.S_init
        pipe_local_s_init = self.pipe_local_s_init
        for i in range(0, branches):
            G_init[i][0] = pipe_area_init[i] * 0.9 * 3600
            abs_G_init[i][i] = abs(G_init[i][0])
            S_init[i][i] = pipe_local_s_init[i]

        # Matrices adjustment
        A_init = self.A_init
        nodes = self.nodes
        flag = max_colume_irrelevant_group(A_init, nodes=nodes, branches=branches)
        new_columes = np.zeros(branches)
        for i in range(0, branches):
            if i < nodes:
                new_columes[i] = flag[i]
            else:
                for j in range(0, branches):
                    for k in range(0, i):
                        stat = 0
                        if new_columes[k] == j:
                            stat = 1
                            break
                    if stat == 0:
                        new_columes[i] = j
                        break
                    else:
                        continue

        # Rearrange calculation matrices
        At = self.At
        Al = self.Al
        A = self.A
        pipe_d = self.pipe_d
        pipe_d_init = self.pipe_d_init
        pipe_area = self.pipe_area
        pipe_length = self.pipe_length
        pipe_length_init = self.pipe_length_init
        pipe_local_s = self.pipe_local_s
        pipe_z = self.pipe_z
        pipe_z_init = self.pipe_z_init
        S = self.S
        G = self.G
        abs_G = self.abs_G
        diff = self.diff
        Gl = self.Gl

        for m in range(0, nodes):
            for n in range(0, branches):
                new = int(new_columes[n])
                if n < nodes:
                    At[m][n] = A_init[m][new]
                else:
                    Al[m][(n - nodes)] = A_init[m][new]
                A[m][n] = A_init[m][new]
        for i in range(0, branches):
            new = int(new_columes[i])
            pipe_d[i] = pipe_d_init[new]
            pipe_area[i] = pipe_area_init[new]
            pipe_length[i] = pipe_length_init[new]
            pipe_local_s[i] = pipe_local_s_init[new]
            pipe_z[i][0] = pipe_z_init[new][0]
            S[i][i] = S_init[new][new]
            G[i][0] = G_init[new][0]
            abs_G[i][i] = abs(G[i][0])
        for j in range(0, diff):
            Gl[j][0] = G[j + nodes][0]
        for k in range(0, branches):
            if A[1][k] == 1:
                self.pump1_branch = k

        Al_T = np.transpose(Al)
        mat = Matrix(At)
        mat_new = mat.rref()
        At_inv = np.linalg.inv(At)
        At_invT = np.transpose(At_inv)
        b = np.dot(Al_T, At_invT)
        Bf = np.zeros((diff, branches))
        for i in range(0, diff):
            for j in range(0, branches):
                if j < nodes:
                    Bf[i][j] = -1 * b[i][j]
                else:
                    if j - nodes == i:
                        Bf[i][j] = 1

        self.At_invT = At_invT
        self.Bf = Bf
        self.new_columes = new_columes

    def get_state_from_datarecorder(self):
        observation = get_latest_observation(self.data_recorder, self.observation_key_dict)
        return observation

    def _snapshot_state(self):
        """Capture a deep copy of the simulator state for temporary rollouts."""
        return copy.deepcopy(self.__dict__)

    def _restore_state(self, snapshot):
        """Restore the simulator to a previously captured state."""
        self.__dict__.clear()
        self.__dict__.update(snapshot)

    def simulate_future_observations(self, action, steps):
        """Simulate `steps` future observations by replaying `action` repeatedly."""
        if steps <= 0:
            return np.empty((0, self.observation_space.shape[0]), dtype=np.float32)

        def to_python_action(value):
            if isinstance(value, np.ndarray):
                if value.ndim == 0:
                    return int(value)
                return value.tolist()
            if isinstance(value, (np.generic, tuple, list)):
                # print(value)
                return value
            return value

        action_input = to_python_action(action)
        # print(action_input)

        snapshot = self._snapshot_state()
        future_obs = []
        try:
            for _ in range(steps):
                next_state, _, done, _, _ = self.step(action_input)
                future_obs.append(np.array(next_state, dtype=np.float32))
                if done:
                    break
        finally:
            self._restore_state(snapshot)

        if not future_obs:
            zero_obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            future_obs.append(zero_obs)
        while len(future_obs) < steps:
            future_obs.append(future_obs[-1].copy())
        return np.stack(future_obs[:steps], axis=0)

    def _get_fcu_power_lookup(self):
        """
        Per-room FCU power lookup for fan levels {0,1,2,3}, aligned with FCU.cal_power():
          level 0 -> 0
          level 1 -> (G_L / G_H)^1.5 * power_basic
          level 2 -> (G_M / G_H)^1.5 * power_basic
          level 3 -> power_basic
        """
        if hasattr(self, "_fcu_power_lookup") and self._fcu_power_lookup is not None:
            return self._fcu_power_lookup

        table = np.zeros((7, 4), dtype=np.float64)
        for r in range(7):
            fcu = self.fcu_list[r]
            g_h = max(float(fcu.G_H), 1e-8)
            table[r, 0] = 0.0
            table[r, 1] = float((fcu.G_L / g_h) ** 1.5 * fcu.power_basic)
            table[r, 2] = float((fcu.G_M / g_h) ** 1.5 * fcu.power_basic)
            table[r, 3] = float(fcu.power_basic)
        self._fcu_power_lookup = table
        return table

    def estimate_rewards_from_observations(self, observations, smoothness: float = 0.0):
        """
        Approximate rewards for a batch of 50-dim observations.

        Note:
        - PPD branch follows get_reward_from_datarecorder's PMV/PPD formulation.
        - Energy term cannot be exactly matched because observation excludes FCU_power;
          we use FCU_fan_feedback as an energy proxy scaled to ~[0, 0.096] per room.
        """
        obs_batch = np.asarray(observations, dtype=np.float64)
        obs_dim = int(np.prod(self.observation_space.shape))
        if obs_batch.ndim == 1:
            obs_batch = obs_batch.reshape(1, -1)
        if obs_batch.ndim != 2 or obs_batch.shape[1] != obs_dim:
            raise ValueError(
                f"observations must be shape (B, {obs_dim}), got {obs_batch.shape}"
            )
        bsz = obs_batch.shape[0]
        constant = self.tradeoff_constant
        room_temp = np.zeros((bsz, 7), dtype=np.float64)
        fan_fb = np.zeros((bsz, 7), dtype=np.float64)
        occupant_activities_num = np.zeros((bsz, 7, 3), dtype=np.float64)
        for r in range(7):
            base = 1 + r * 7
            room_temp[:, r] = obs_batch[:, base]
            fan_fb[:, r] = obs_batch[:, base + 1]
            occupant_activities_num[:, r, :] = obs_batch[:, base + 4: base + 7]

        if not hasattr(self, "pmvppd_lookup"):
            self.construct_pmvppd_lookup()

        tdb = np.clip(
            room_temp,
            self.pmvppd_lookup.tdb_range[0],
            self.pmvppd_lookup.tdb_range[-1],
        )
        tdb_idx = np.searchsorted(self.pmvppd_lookup.tdb_range, tdb)
        tdb_idx = np.clip(tdb_idx, 0, len(self.pmvppd_lookup.tdb_range) - 1)
        ppd_matrix = np.zeros((bsz, 7, 3), dtype=np.float64)
        for act in range(3):
            ppd_matrix[:, :, act] = self.pmvppd_lookup.ppd_table[act, tdb_idx]

        total_occupant_num = np.sum(occupant_activities_num, axis=(1, 2))
        total_ppd = np.sum(ppd_matrix * occupant_activities_num, axis=(1, 2))
        mean_ppd = np.divide(
            total_ppd,
            total_occupant_num,
            out=np.zeros_like(total_ppd),
            where=total_occupant_num > 0,
        )

        # Align with get_energy_consumption_from_datarecorder as much as possible:
        # approximate FCU_power from fan feedback via the same fan-power curve used in FCU.cal_power.
        # Note: observation has no onoff_set, so this assumes fan_feedback=0 implies off.
        power_lookup = self._get_fcu_power_lookup()  # (7, 4)
        fan_level = np.clip(np.rint(fan_fb), 0, 3).astype(np.int64)  # (B, 7)
        room_idx = np.arange(7)[None, :]
        room_power = power_lookup[room_idx, fan_level]  # (B, 7)
        energy_proxy = np.sum(room_power, axis=1)

        if self.reward_mode == "Baseline_without_energy":
            r_t = (room_temp - 25.0) ** 2
            temperature_bias = np.sum(r_t, axis=1)
            return -temperature_bias
        if self.reward_mode == "Baseline_with_energy":
            t_up, t_low = 25.0, 23.0
            r_t = np.abs(room_temp - t_up) + np.abs(room_temp - t_low) - np.abs(t_up - t_low)
            temperature_bias = np.sum(r_t, axis=1)
            return -temperature_bias - constant * energy_proxy
        if self.reward_mode == "Baseline_OCC_PPD_without_energy":
            return -mean_ppd
        if self.reward_mode == "Baseline_OCC_PPD_with_energy":
            return -mean_ppd - constant * energy_proxy - 3.0 * smoothness
        if self.reward_mode == "Baseline_occ_with_energy_new":
            return -mean_ppd / 100.0 - 0.2 * energy_proxy / 0.672
        if self.reward_mode == "Baseline_occ_with_energy_sm":
            return -mean_ppd / 100.0 - 0.2 * energy_proxy / 0.672 - self.smoothness_constant * smoothness
        raise NotImplementedError(
            f"estimate_reward_from_observations not implemented for reward_mode={self.reward_mode!r}"
        )

    def estimate_reward_from_observation(self, obs, smoothness: float = 0.0):
        rewards = self.estimate_rewards_from_observations(obs, smoothness=smoothness)
        return float(np.asarray(rewards).reshape(-1)[0])

    def get_reward_from_datarecorder(self):
        constant = self.tradeoff_constant

        if self.reward_mode == "Baseline_without_energy":
            temperature_bias = self.get_temperature_bias_from_datarecorder_2nd()
            reward = - temperature_bias
        elif self.reward_mode == "Baseline_with_energy":
            temperature_bias = self.get_temperature_bias_from_datarecorder()
            energy_consumption = self.get_energy_consumption_from_datarecorder()
            reward = - temperature_bias - constant * energy_consumption
        elif self.reward_mode == "Baseline_OCC_PPD_without_energy":
            mean_ppd = self.get_ppd_from_datarecorder()
            reward = - mean_ppd
        elif self.reward_mode == "Baseline_OCC_PPD_with_energy":
            mean_ppd = self.get_ppd_from_datarecorder()
            energy_consumption = self.get_energy_consumption_from_datarecorder()
            reward = - mean_ppd - constant * energy_consumption
            # smoothness_cost = 5 * np.log1p(self.current_step_smoothness)
            # reward -= smoothness_cost
            # smoothness_cost = 3 * self.current_step_smoothness
            # reward -= smoothness_cost
        elif self.reward_mode == "Baseline_occ_with_energy_new":
            mean_ppd = self.get_ppd_from_datarecorder()
            energy_consumption = self.get_energy_consumption_from_datarecorder()
            reward = - mean_ppd/100 - 0.2 * energy_consumption/0.672
        elif self.reward_mode == "Baseline_occ_with_energy_sm":
            mean_ppd = self.get_ppd_from_datarecorder()
            energy_consumption = self.get_energy_consumption_from_datarecorder()
            reward = - mean_ppd/100 - 0.2 * energy_consumption/0.672
            smoothness_cost = self.smoothness_constant * getattr(self, "current_step_smoothness", 0.0)
            reward -= smoothness_cost

        return reward

    def get_temperature_bias_from_datarecorder(self, T_up=25, T_low=23):
        # Reference: https://ugr-sail.github.io/sinergym/compilation/main/pages/rewards.html
        room_temp = get_latest_observation_from_every_room(self.data_recorder, "room_temp")
        r_t = np.abs(room_temp - T_up) + np.abs(room_temp - T_low) - np.abs(T_up - T_low)
        # Sum of the biases in every room, temperature_bias > 0, 越接近0越好, 越小越好
        temperature_bias = np.sum(r_t)
        return temperature_bias

    def get_temperature_bias_from_datarecorder_2nd(self, T_set = 25):
        # For YangXu Project
        room_temp = get_latest_observation_from_every_room(self.data_recorder, "room_temp")
        r_t = (room_temp - T_set)**2
        # Sum of the biases in every room, temperature_bias > 0, 越接近0越好, 越小越好
        temperature_bias = np.sum(r_t)
        return temperature_bias

    def get_energy_consumption_from_datarecorder(self):
        energy_vec = get_latest_observation_from_every_room(self.data_recorder, "FCU_power")
        total_energy_consumption = np.sum(energy_vec) # 总能耗 范围：0.0 - 0.096*7即0.0 - 0.672
        # total_energy_consumption > 0, 越接近0越好
        return total_energy_consumption

    def get_pmv_ppd_from_datarecorder(self):
        # 判断self是否包含pmvppd_lookup，如果没有，则构造
        if not hasattr(self, 'pmvppd_lookup'):
            self.construct_pmvppd_lookup()

        occupant_num_vec = get_latest_observation_from_every_room(self.data_recorder, "occupant_num")
        occupant_activities_num = occupant_num_vec.reshape(7, 3)

        room_temp = get_latest_observation_from_every_room(self.data_recorder, "room_temp")

        # from pythermalcomfort.models import pmv_ppd
        pmv_matrix = np.zeros_like(occupant_activities_num, dtype=float)
        ppd_matrix = np.zeros_like(occupant_activities_num, dtype=float)

        # Calculate PMV and PPD for each room and each occupant activity
        for i in range(7):
            for j in range(3):
                tdb = room_temp[i]
                # vr = vr_activity[j]
                # met = met_activity[j]
                # clo = clo_activity[j]
                # results = pmv_ppd(tdb=tdb, tr=tdb, vr=vr, rh=rh, met=met, clo=clo, standard="ASHRAE")
                # pmv_matrix[i][j] = results['pmv']
                # ppd_matrix[i][j] = results['ppd']

                pmv, ppd = self.pmvppd_lookup.query(tdb, j)
                pmv_matrix[i][j] = pmv
                ppd_matrix[i][j] = ppd

        total_occupant_num = np.sum(occupant_num_vec)

        # pmv 冷(-3)//凉(-2)/稍凉(-1)//中性(0)//稍暖(+1)//暖(+2)//热(+3)
        pmv_matrix_abs = np.abs(pmv_matrix)
        total_pmv = np.sum(pmv_matrix_abs * occupant_activities_num)

        total_ppd = np.sum(ppd_matrix * occupant_activities_num)

        if total_occupant_num == 0:
            mean_pmv = 0.0
            mean_ppd = 0.0
        else:
            # 平均pmv_abs越小（越接近0）越好
            mean_pmv = total_pmv / total_occupant_num
            # 平均ppd越小（越接近0）越好
            mean_ppd = total_ppd / total_occupant_num

        return mean_pmv, mean_ppd

    def get_pmv_from_datarecorder(self):
        mean_pmv, _ = self.get_pmv_ppd_from_datarecorder()
        return mean_pmv

    def get_ppd_from_datarecorder(self):
        _, mean_ppd = self.get_pmv_ppd_from_datarecorder()
        return mean_ppd

    def construct_pmvppd_lookup(self):
        # 构造PMV-PPD lookup表
        # Sitting 1, walking 2, standing 3

        # vr_sitting = 0.15 vr_walking = 0.45 vr_standing = 0.27
        vr_activity = [0.15, 0.45, 0.27]
        # met_sitting = 1.0 met_walking = 2.0 met_standing = 1.4
        met_activity = [1.0, 2.0, 1.4]
        # clo_sitting = 0.63 clo_walking = 0.504 clo_standing = 0.558
        clo_activity = [0.63, 0.504, 0.558]
        # 湿度
        rh = 40
        # 温度上下限，避免仿真器的问题
        tdb_up_bound = 38
        tdb_low_bound = 10

        self.pmvppd_lookup = SimplifiedPMVPPDLookup(tdb_low_bound=tdb_low_bound,
                                                    tdb_up_bound=tdb_up_bound,
                                                    vr_activity=vr_activity,
                                                    met_activity=met_activity,
                                                    clo_activity=clo_activity,
                                                    rh=rh)


if __name__ == "__main__":
    config_path = 'SemiPhysBuildingSim/hyperparams/spbs_default.yml'
    # # 读取YAML文件并加载为EasyDict并转换所有列表为 NumPy 数组
    # with open(config_path, 'r') as file:
    #     params = EasyDict(yaml.safe_load(file))
    # params = convert_lists_to_np_arrays(params=params)
    #
    # env0 = SemiPhysBuildingSimulation(hyperparams=params)

    env1 = SemiPhysBuildingSimulation(hyperparams_path=config_path)
    # obs = env1.reset()
    # rewards = 0
    # done = False
    # i = 0

    for _ in range(1):
        obs = env1.reset()
        rewards = 0
        done = False
        i = 0
        while not done:
            # print(str(i) + "th obs: " + str(obs))
            i += 1
            # action = [3, 3, 3, 3, 3, 3, 3]  #
            # action = [0,0,0, 0, 0, 0, 0]#[3,3,3, 3, 3, 3, 3]#env1.action_space.sample()
            action = 0
            print(action)
            obs, r, done,_, info = env1.step(action)
            # print("action:"+str(action))
            # if i >= 100:
            #     break

            rewards += r
        print("rewards:" + str(rewards))
    data_recorder = env1.data_recorder
    import matplotlib.pyplot as plt

    # Data to plot
    data = data_recorder["room1"]["room_temp"]
    data2 = data_recorder["sensor_outdoor"]["outdoor_temp"]
    data3 = data_recorder["training"]["reward"]

    # 创建包含两个子图的图形，垂直排列
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 第一个子图：Temperature 和 Outdoor Temperature
    ax1.plot(data, marker='o', linestyle='-', color='b', label='Temperature')
    ax1.plot(data2, marker='o', linestyle='-', color='r', label='Outdoor Temperature')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Value')
    ax1.set_title('Temperature Over Time')
    ax1.legend()
    ax1.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # 第二个子图：Reward
    ax2.plot(data3, marker='o', linestyle='-', color='g', label='Reward')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Reward')
    ax2.set_title('Reward Over Time')
    ax2.legend()
    ax2.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # 调整子图间距
    plt.tight_layout()

    # 保存图形
    output_path = Path(__file__).with_name("simulation_result.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    # 显示图形
    plt.show()
