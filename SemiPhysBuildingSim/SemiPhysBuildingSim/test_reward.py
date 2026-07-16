import numpy as np

def reward_function(room_temp):
    T_up = 25
    T_low = 23
    r_t = np.abs(room_temp - T_up) + np.abs(room_temp - T_low) - np.abs(T_up - T_low)
    reward = -np.sum(r_t)
    return reward


# plot the reward function for different room temperatures
import matplotlib.pyplot as plt
room_temp = np.arange(10, 40, 0.5)
for i in range(len(room_temp)):
    reward = reward_function(room_temp[i])
    plt.plot(room_temp[i], reward, 'o')
plt.xlabel('Room Temperature')
plt.ylabel('Reward')
plt.show()

