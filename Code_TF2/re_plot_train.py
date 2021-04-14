import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

walking_style = "three_legged_walk"
model_name = "pb_pzpos_linear_prove_0"
average_return = "avg_return1000998.npy"
performance = "performance1000998.npy"

# Re plot training with transparent blue
avg_ret = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                  "/{}/{}/{}".format(walking_style, model_name, average_return))
perfor = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                 "/{}/{}/{}".format(walking_style, model_name, performance))

fig_new, ax_new = plt.subplots()
ax_new.plot(range(len(perfor)), perfor, 'b', alpha=0.5, label="(a)")
ax_new.plot(range(len(avg_ret)), avg_ret, color='r', label="(b)")
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
plt.legend(loc="upper left")
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/a_episodes_training_fig.pdf".format(walking_style, model_name), bbox_inches='tight')

