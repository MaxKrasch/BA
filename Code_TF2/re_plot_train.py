import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

walking_style = "positiv_zpos_walk"
model_name = "pb_pzpos_prove_4"
average_return = "avg_return1000584.npy"
performance = "performance1000584.npy"

# Re plot training with transparent blue
avg_ret = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                  "/{}/{}/{}".format(walking_style, model_name, average_return))
perfor = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                 "/{}/{}/{}".format(walking_style, model_name, performance))

fig_new, ax_new = plt.subplots()
ax_new.plot(range(len(avg_ret)), avg_ret, color='r')
ax_new.plot(range(len(perfor)), perfor, 'b', alpha=0.5)
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/a_episodes_training_fig.pdf".format(walking_style, model_name), bbox_inches='tight')
