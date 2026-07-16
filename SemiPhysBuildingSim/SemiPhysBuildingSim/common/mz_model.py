import math
import numpy as np
import csv

import numpy as np
import csv

class ZONE():
    def __init__(self, name, area, occ_list_length,hyperparams, Itd):
        # Constants
        self.c = hyperparams.reference.c
        self.alpha = hyperparams.reference.alpha
        self.Rsor = hyperparams.reference.Rsor
        self.Rsow = hyperparams.reference.Rsow
        self.Uw = hyperparams.reference.Uw
        self.Uf = hyperparams.reference.Uf
        self.Sc = hyperparams.reference.Sc
        self.WWR = hyperparams.reference.WWR
        self.qp = hyperparams.reference.qp
        self.qd = hyperparams.reference.qd
        self.Itd = Itd
        # Basic
        self.name = name
        self.area = area         #  m2
        self.volume = area * 3   #  m3
        self.temp = 26.5 #16  #  oC
        self.temp_set = 26.5
        self.RH = 40.0
        self.RH_set = 70.0
        self.d = 5.11    #  g/kg(dry air)
        self.p = 15.0      #  Relative Pressure
        self.leakage_area = 0.0   #  m2
        self.wall_area = [0, 0, 0, 0]   #  N, W, S, E, Roof
        self.rou = 1.293 * (273.15 / (self.temp + 273.15))
        self.sup_temp = 14 #40
        self.sup_airflow = 0.         #  m3/min
        self.return_airflow = 0.      #  m3/min
        self.fan_position = 0
        self.exfiltration_ariflow = 0.
        self.transferIn_airflow = 0.
        self.transferOut_airflow = 0.
        # Temp
        self.DT = 0.
        self.TDeqr = 0.
        self.TDeqw = np.zeros(4)
        self.Qw = 0.
        self.Qe = 0.
        self.Qp = 0.
        self.Q_load = 0.
        self.Qa = 0.
        self.Q_trans = 0.
        # Occupant

        self.occupant_trans = 0
        self.occupant_list = np.zeros(occ_list_length)
        self.occupant_trans_list = np.zeros(occ_list_length)

        self.use_honeycomb = False

        if self.use_honeycomb:
            self.occupant_num = 0
        else:
            self.occupant_num = {"sitting": 0, "walking": 0, "standing": 0}

        if self.use_honeycomb:
            self.occupant_num_data = []
            path = f"SemiPhysBuildingSim/honeycomb_data/room_{self.name[4]}_result.csv"
            with open(path) as csv_file:
                csv_reader = csv.reader(csv_file)
                next(csv_reader)
                for row in csv_reader:
                    self.occupant_num_data.append(int(row[1]))
        else:
            # if not use honeycomb, we will use data from human_activity_data.csv
            sheet_dict = {'room1': 'Sheet1', 'room2': 'Sheet2', 'room3': 'Sheet3', 'room4': 'Sheet1',
                          'room5': 'Sheet2', 'room6': 'Sheet3', 'room7': 'Sheet1', 'room8': 'Sheet2', }

            path = f"SemiPhysBuildingSim/human_activity_data/final_with_min_sampled_manual.xlsx"
            # path = f"SemiPhysBuildingSim/human_activity_data/generated_winter_person.xlsx"
            # 读取 Excel 文件
            import pandas as pd
            df = pd.read_excel(path, sheet_name=sheet_dict[self.name])

            self.sitting_list = df['sitting'].tolist()
            self.walking_list = df['walking'].tolist()
            self.standing_list = df['standing'].tolist()

            # # total_occupants = sitting + walking + standing
            # self.occupant_num_data = [sitting_list[i] + walking_list[i] + standing_list[i] for i in
            #                          range(len(sitting_list))]
        # Alarm
        self.temp_alarm = 0
        self.RH_alarm = 0
        self.pandemic_alarm = 0

        # Nano
        self.nano_flag = False
        self.history_flag = True

    def cal_room(self, dry_bulb_t, min, rou, G_fan, all_G_fan):  # min
        # self.occupant_num = self.occupant_num_data[min+541]
        # self.occupant_num = random.randint(0, 8)
        # Constants
        c = self.c
        alpha = self.alpha
        Rsor = self.Rsor
        Rsow = self.Rsow
        Uw = self.Uw
        Uf = self.Uf
        Sc = self.Sc
        WWR = self.WWR
        qp = self.qp
        qd = self.qd
        Itd = self.Itd

        if self.use_honeycomb:
            occupant_num = self.occupant_num
        else:
            occupant_num = sum(self.occupant_num.values())
        self.occupant_trans = occupant_num - self.occupant_list[-1]
        for o in range(0, len(self.occupant_list)):
            if o < (len(self.occupant_list) - 1):
                self.occupant_list[o] = self.occupant_list[o+1]
                self.occupant_trans_list[o] = self.occupant_trans_list[o+1]
            else:
                self.occupant_list[o] = occupant_num
                self.occupant_trans_list[o] = self.occupant_trans

        # print("all_G_fan:"+str(all_G_fan))
        # print("name:" + self.name + " G_fan:" + str(G_fan))

        G_fan_origin = G_fan
        if self.name == "room4":
            G_fan = 0.5 * G_fan_origin + 0.3*all_G_fan[4] + 0.2*all_G_fan[5]
        if self.name == "room5":
            G_fan = 0.5 * G_fan_origin + 0.25*all_G_fan[3] + 0.25*all_G_fan[5]
        if self.name == "room6":
            G_fan = 0.5 * G_fan_origin + 0.2*all_G_fan[3] + 0.3*all_G_fan[4]


        self.Qa_0 = c * rou * G_fan/60 * (self.sup_temp - self.temp)
        # if self.name == "room4":
        #     self.Qa = 0.5 * self.Qa_0 + 0.3*Qa_last_minute[4] + 0.2*Qa_last_minute[5]
        # if self.name == "room5":
        #     self.Qa = 0.5 * self.Qa_0 + 0.25*Qa_last_minute[3] + 0.25*Qa_last_minute[5]
        # if self.name == "room6":
        #     self.Qa = 0.5 * self.Qa_0 + 0.1*Qa_last_minute[3] + 0.3*Qa_last_minute[4]
        self.Qa = self.Qa_0
        self.Qw = 0
        self.DT = dry_bulb_t - self.temp
        self.TDeqr = self.DT + (alpha * Rsor * Itd[min+540][4])
        f = 0.3    # Factor of envelope
        for i in range (0, 4):
            self.TDeqw[i] = self.DT + (alpha * Rsow * Itd[min+540][i])
        for j in range (0, 4):
            self.Qw += self.wall_area[j] * ((1-WWR)*self.TDeqw[j]*Uw+WWR*self.DT*Uf+WWR*Sc*Itd[min+540][j])
        self.Qe = f * self.Qw
        # TODO in 11/01: for human activity
        # 60 * 显热功率替换qp
        self.Qp = 60 * occupant_num * (qp*((37.0 - self.temp)/(37.0 - 24.0)) + qd)

        self.Q_load = self.Qe + self.Qp              #  kJ/min
        self.temp += 0.5 * (self.Q_load + self.Qa) / c / self.rou / self.volume

    def cal_damp(self):
        pass



class FCU():
    def __init__(self, FCU_name, G_H, G_M, G_L, Qh_H, Qh_M, Qh_L, Qc_H, Qc_M, Qc_L, Power_basic, waterflow_basic,
                 hyperparams):    # 初始化为样本额定工况参数
        self.c = hyperparams.reference.c
        self.FCU_name = FCU_name
        self.G_H = G_H           # air flow,  m3/h
        self.G_M = G_M
        self.G_L = G_L
        self.Qh_H = Qh_H / 1000  # Q heat,  kw
        self.Qh_M = Qh_M / 1000
        self.Qh_L = Qh_L / 1000
        self.Qc_H = Qc_H / 1000  # Q cooling,  kw
        self.Qc_M = Qc_M / 1000
        self.Qc_L = Qc_L / 1000
        self.mode_set = 1
        self.mode = 1            # 1-cooling   2-heating
        self.power_basic = Power_basic / 1000           # fan power at H,  kw
        self.waterflow_basic = waterflow_basic / 1000   #  m3/h
        self.Qh_tw_supply = 60
        self.Qh_tw_return = 55
        self.Qc_tw_supply = 7
        self.Qc_tw_return = 12

        self.onoff = 0
        self.onoff_set = 0
        self.fan_position = 0
        self.fan_set = 0
        self.G_fan = 0.
        self.Q = 0.
        self.waterflow = 0.
        self.power = 0.
        self.tw_supply = 7
        self.tw_return = 12
        self.tw_supplypressure = 0
        self.tw_returnpressure = 0
        self.valve_position = 100
        self.valve_set = 100

        # Alarm
        self.flow_alarm = 0
        self.tw_alarm = 0
        self.pressure_alarm = 0

        c = self.c
        self.eta_Qh_H = self.Qh_H / (c * 1000 * self.waterflow_basic * (self.Qh_tw_supply - self.Qh_tw_return))
        self.eta_Qh_M = self.Qh_M / (c * 1000 * self.waterflow_basic * (self.Qh_tw_supply - self.Qh_tw_return))
        self.eta_Qh_L = self.Qh_L / (c * 1000 * self.waterflow_basic * (self.Qh_tw_supply - self.Qh_tw_return))
        self.eta_Qc_H = self.Qc_H / (c * 1000 * self.waterflow_basic * (self.Qc_tw_supply - self.Qc_tw_return))
        self.eta_Qc_M = self.Qc_M / (c * 1000 * self.waterflow_basic * (self.Qc_tw_supply - self.Qc_tw_return))
        self.eta_Qc_L = self.Qc_L / (c * 1000 * self.waterflow_basic * (self.Qc_tw_supply - self.Qc_tw_return))

    def cal_fan_G(self):
        if self.onoff_set == 0:
            self.onoff = 0
            self.G_fan = 0
        if self.onoff_set == 1:
            self.onoff = 1
            if self.fan_position == 1:
                self.G_fan = self.G_L
            elif self.fan_position == 2:
                self.G_fan = self.G_M
            elif self.fan_position == 3:
                self.G_fan = self.G_H
            else:
                self.onoff = 0
                self.G_fan = 0

    def cal_supplyair(self,room_t):
        if self.mode == 1:
            supply_t = 20 - 30 * math.log((self.waterflow + 1) * math.e / 2)
        if self.mode == 2:
            supply_t = 30 + 8 * math.log((self.waterflow + 1) * math.e / 2)
        return supply_t

    def cal_returntemp(self, Q):
        c = self.c
        if self.mode == 1:
            if self.fan_position == 0:
                self.tw_return = self.tw_supply
            if self.fan_position == 1:
                self.tw_return = self.tw_supply + Q / (self.eta_Qc_L * c * 1000 * self.waterflow)
            if self.fan_position == 2:
                self.tw_return = self.tw_supply + Q / (self.eta_Qc_M * c * 1000 * self.waterflow)
            if self.fan_position == 3:
                self.tw_return = self.tw_supply + Q / (self.eta_Qc_H * c * 1000 * self.waterflow)
        if self.mode == 2:
            if self.fan_position == 0:
                self.tw_return = self.tw_supply
            if self.fan_position == 1:
                self.tw_return = self.tw_supply - Q / (self.eta_Qh_L * c * 1000 * self.waterflow)
            if self.fan_position == 2:
                self.tw_return = self.tw_supply - Q / (self.eta_Qh_M * c * 1000 * self.waterflow)
            if self.fan_position == 3:
                self.tw_return = self.tw_supply - Q / (self.eta_Qh_H * c * 1000 * self.waterflow)

    def cal_power(self):
        if self.fan_position == 0 or self.onoff_set == 0:
            self.power = 0
        if self.fan_position == 1:
            self.power = pow((self.G_L / self.G_H), 1.5) * self.power_basic
        if self.fan_position == 2:
            self.power = pow((self.G_M / self.G_H), 1.5) * self.power_basic
        if self.fan_position == 3:
            self.power = self.power_basic


class PUMP():
    def __init__(self, name, hyperparams):
        self.PI = hyperparams.reference.PI
        self.name = name
        self.n_basic = 50
        self.n = 40
        self.n_set = 40
        self.onoff = 0
        self.onoff_set = 0
        self.valve_position = 0
        self.valve_set = 0
        self.s = 10.
        PI = self.PI
        self.G = PI/4*0.065*0.065 * 3600               # m3/h
        self.H = 0.
        self.a_18 = -0.007053
        self.b_18 = 0.09987
        self.c_18 = 19.81
        self.a_24 = -0.005427
        self.b_24 = 0.06342
        self.c_24 = 25.67
        self.power_basic18 = 2.2  # kw
        self.power_basic24 = 3.0  # kw

        self.p_supply = 3 * 9800
        self.p_return = 3 * 9800
        self.supply_temp = 7.0
        self.return_temp = 12.0

        # Alarm
        self.flow_alarm = 0
        self.temp_alarm = 0
        self.pressure_alarm = 0
        self.head_alarm = 0

    def cal_pump(self):
        if self.onoff == 0:
            self.H = 0
            self.dH_dG = 0
            self.power = 0
        if self.onoff == 1:
            # 18
            # self.H_basic = (self.a_18 * math.pow(self.G, 2) + self.b_18 * self.G + self.c_18) * 9800
            # self.H = math.pow((self.n / self.n_basic), 2) * self.H_basic
            # self.dH_dG = math.pow((self.n / self.n_basic), 2) * (2 * self.a_18 * self.G + self.b_18) * 9800
            # self.power = math.pow((self.n / self.n_basic), 3) * self.power_basic18
            # 24
            self.H_basic = (self.a_24 * math.pow(self.G, 2) + self.b_24 * self.G + self.c_24) * 9800
            self.H = math.pow((self.n / self.n_basic), 2) * self.H_basic
            self.dH_dG = math.pow((self.n / self.n_basic), 2) * (2 * self.a_24 * self.G + self.b_24) * 9800
            self.power = math.pow((self.n / self.n_basic), 3) * self.power_basic24
            self.p_supply = self.p_return + self.H

        if self.valve_position == 1:
            self.s = 10.
        if self.valve_position == 0:
            self.s = 10000000


class HEATPUMP():
    def __init__(self):
        self.name = 'heatpump'
        self.onoff = 0
        self.mode = 2
        self.supplytemp = 60.0
        self.supply_tempset = 60.0

        # ALarm
        self.flow_alarm = 0
        self.temp_alarm = 0
        self.pressure_alarm = 0