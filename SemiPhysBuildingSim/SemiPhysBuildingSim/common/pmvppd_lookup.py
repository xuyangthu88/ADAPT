import numpy as np
from pythermalcomfort.models import pmv_ppd


class SimplifiedPMVPPDLookup:
    def __init__(self,
                 vr_activity=[0.15, 0.45, 0.27],
                 met_activity=[1.0, 2.0, 1.4],
                 clo_activity=[0.63, 0.504, 0.558],
                 rh=40,
                 resolution=50,
                 tdb_low_bound = 10,
                 tdb_up_bound = 38,):
        # # Sitting 1, walking 2, standing 3
        #
        # # vr_sitting = 0.15 vr_walking = 0.45 vr_standing = 0.27
        # vr_activity = [0.15, 0.45, 0.27]
        # # met_sitting = 1.0 met_walking = 2.0 met_standing = 1.4
        # met_activity = [1.0, 2.0, 1.4]
        # # clo_sitting = 0.63 clo_walking = 0.504 clo_standing = 0.558
        # clo_activity = [0.63, 0.504, 0.558]
        # # 湿度
        # rh = 40
        # # 温度上下限，避免仿真器的问题
        # tdb_up_bound = 38
        # tdb_low_bound = 10

        self.tdb_low_bound = tdb_low_bound
        self.tdb_up_bound = tdb_up_bound
        self.tdb_range = np.linspace(tdb_low_bound, tdb_up_bound, resolution)
        self.vr_activity = vr_activity
        self.met_activity = met_activity
        self.clo_activity = clo_activity
        self.rh = rh

        # 初始化查找表: 每种 (vr, met, clo) 对应一张表
        self.pmv_table = np.zeros((len(vr_activity), resolution))
        self.ppd_table = np.zeros((len(vr_activity), resolution))

        # 生成查找表
        for idx, (vr, met, clo) in enumerate(zip(vr_activity, met_activity, clo_activity)):
            for i, tdb in enumerate(self.tdb_range):
                results = pmv_ppd(tdb=tdb, tr=tdb, vr=vr, rh=self.rh, met=met, clo=clo, standard="ASHRAE")
                self.pmv_table[idx, i] = results["pmv"]
                self.ppd_table[idx, i] = results["ppd"]

    def query(self, tdb, activity_idx):
        # 将 tdb 限制在查表范围内
        tdb = np.clip(tdb, self.tdb_range[0], self.tdb_range[-1])
        # 查找对应温度的 PMV 和 PPD
        idx = np.searchsorted(self.tdb_range, tdb)  # 找到最接近的索引
        idx = min(idx, len(self.tdb_range) - 1)  # 防止越界
        return self.pmv_table[activity_idx, idx], self.ppd_table[activity_idx, idx]


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    lookup = SimplifiedPMVPPDLookup(tdb_low_bound=13,tdb_up_bound=35)

    plt.figure(figsize=(14, 6))
    tdb_values = lookup.tdb_range

    for i, activity in enumerate(["Sitting", "Walking", "Standing"]):
        # pmv_values = lookup.pmv_table[i]
        # ppd_values = lookup.ppd_table[i]
        pmv_values, ppd_values = [], []
        for tdb in tdb_values:
            pmv, ppd = lookup.query(tdb, i)
            pmv_values.append(pmv)
            ppd_values.append(ppd)

        plt.subplot(1, 2, 1)
        plt.plot(tdb_values, pmv_values,
                 label=f"{activity}")
                 # label=f"{activity} (vr={lookup.vr_activity[i]}, met={lookup.met_activity[i]}, clo={lookup.clo_activity[i]})")
        plt.title("PMV under Different Activities",fontsize=15)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("PMV")

        plt.grid(True, which='both', linewidth=0.5, alpha=0.5)
        plt.xticks(np.arange(lookup.tdb_low_bound, lookup.tdb_up_bound + 1, 1))  # Ensure grid at each degree
        plt.yticks(np.arange(-5, 5.1, 1))  # Ensure grid at each degree
    plt.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label="Thermal Comfort Zone")
    plt.axhline(y=-0.5, color='red', linestyle='--', linewidth=2)
    plt.tick_params(labelsize=10)

    handles, labels = plt.gca().get_legend_handles_labels()
    new_order = [0, 2, 1,3]  # 按照新的顺序重新排列handles和labels
    plt.legend([handles[i] for i in new_order], [labels[i] for i in new_order],loc='upper left',prop={'size': 15})

    for i, activity in enumerate(["Sitting", "Walking", "Standing"]):
        pmv_values, ppd_values = [], []
        for tdb in tdb_values:
            pmv, ppd = lookup.query(tdb, i)
            pmv_values.append(pmv)
            ppd_values.append(ppd)
        plt.subplot(1, 2, 2)
        plt.plot(tdb_values, ppd_values,
                 label=f"{activity} (vr={lookup.vr_activity[i]}, met={lookup.met_activity[i]}, clo={lookup.clo_activity[i]})")
        plt.title("PPD under Different Activities",fontsize=15)
        plt.xlabel("Temperature (°C)")
        plt.ylabel("PPD (%)")
        # plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.xticks(np.arange(lookup.tdb_low_bound, lookup.tdb_up_bound + 1, 1))  # Ensure grid at each degree
        plt.yticks(np.arange(0, 101, 10))  # Ensure grid at each degree
    plt.axhline(y=10, color='red', linestyle='--', linewidth=2)
    plt.tick_params(labelsize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig("pmv_ppd_lookup.pdf", dpi=300)