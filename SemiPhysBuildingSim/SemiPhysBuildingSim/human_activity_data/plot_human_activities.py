import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取 Excel 文件
df = pd.read_excel('generated_winter_person.xlsx', sheet_name='Sheet3')

# 提取各状态人数列
sitting_list = df['sitting'].tolist()
walking_list = df['walking'].tolist()
standing_list = df['standing'].tolist()

# 总人数 = 坐着 + 站着 + 走动
total_occupants_list = [sitting_list[i] + walking_list[i] + standing_list[i] for i in range(len(sitting_list))]


sitting_list = np.array(sitting_list)
walking_list = np.array(walking_list)
standing_list = np.array(standing_list)
total_occupants_list = np.array(total_occupants_list)


# 每15分钟一个数据点，总共需要的数据长度是根据间隔来推算
index_time = np.arange(0, 481, 15)  # 按照实际数据长度调整
time = index_time

occupancy_sitting = sitting_list[index_time]  # 每15分钟选择一个坐着的人数
occupancy_standing = standing_list[index_time]  # 每15分钟选择一个站着的人数
occupancy_walking = walking_list[index_time]  # 每15分钟选择一个走动的人数
occupancy_total = total_occupants_list[index_time]  # 每15分钟选择一个总人数

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 10  # 柱子宽度

# 绘制堆叠条形图
ax.bar(time, occupancy_sitting, label='Sitting', color='skyblue', width=bar_width)
ax.bar(time, occupancy_standing, bottom=occupancy_sitting, label='Standing', color='orange', width=bar_width)
ax.bar(time, occupancy_walking, bottom=occupancy_sitting + occupancy_standing, label='Walking', color='green', width=bar_width)

# 绘制总人数曲线
# ax.plot(time, occupancy_total, label='Total', color='black', linewidth=2)

ax.plot(time, occupancy_total, label='Total', color='black')

# 设置第一个子图的标签和标题
# ax.set_xlabel('Time (Minutes)')
ax.set_ylabel('Occupancy')
# ax.set_title('Occupancy Over Time')
ax.legend(loc='upper left',ncol=4)

ax.set_ylim(0, 5)  # Set y-axis limits for occupancy
ax.yaxis.set_ticks(range(0, 5, 1))  # Set y-ticks for occupancy

ax.set_xlim(-20, 500)  # Set x-axis limits
ax.xaxis.set_ticks(range(0, 481, 60))  # Set x-ticks at intervals of 60
time_labels = ['9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00']
ax.set_xticklabels(time_labels)  # Set x-tick labels
ax.grid(True)

plt.show()
