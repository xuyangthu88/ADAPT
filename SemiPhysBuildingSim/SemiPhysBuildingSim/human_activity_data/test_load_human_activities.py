import pandas as pd

# 读取 Excel 文件
df = pd.read_excel('final_with_min_sampled_manual.xlsx', sheet_name='Sheet1')

sitting_list = df['sitting'].tolist()
walking_list = df['walking'].tolist()
standing_list = df['standing'].tolist()

# total_occupants = sitting + walking + standing
total_occupannts_list = [sitting_list[i] + walking_list[i] + standing_list[i] for i in range(len(sitting_list))]

print(total_occupannts_list)