import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# Re plot with transparent blue
avg_ret = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
                  "/pb1_ground_only_simple/avg_return1000109.npy")
perfor = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
                 "/pb1_ground_only_simple/performance1000109.npy")

fig_new, ax_new = plt.subplots()
ax_new.plot(range(len(avg_ret)), avg_ret, color='r')
ax_new.plot(range(len(perfor)), perfor, 'b', alpha=0.5)
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
            "/pb1_ground_only_simple/gallop_training_fig.pdf", bbox_inches='tight')
