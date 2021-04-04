import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

# Re plot training with transparent blue
# avg_ret = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
#                   "/pb1_ground_only_simple/avg_return1000109.npy")
# perfor = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
#                  "/pb1_ground_only_simple/performance1000109.npy")
#
# fig_new, ax_new = plt.subplots()
# ax_new.plot(range(len(avg_ret)), avg_ret, color='r')
# ax_new.plot(range(len(perfor)), perfor, 'b', alpha=0.5)
# plt.xlabel("Episodes")
# plt.ylabel("Performance")
# plt.grid(True)
# # plt.show()
# plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves/hopper_walk"
#             "/pb1_ground_only_simple/gallop_training_fig.pdf", bbox_inches='tight')


# Walking Style Plots

walking_style = "normal_walk"

electricity_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/electricity_list_0.npy".format(walking_style))
electricity_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/electricity_list_1.npy".format(walking_style))
electricity_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/electricity_list_2.npy".format(walking_style))
electricity_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/electricity_list_3.npy".format(walking_style))
electricity_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/electricity_list_4.npy".format(walking_style))
electricity_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                 "/{}/np_arrays/electricity_std_list_0.npy".format(walking_style))
electricity_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                 "/{}/np_arrays/electricity_std_list_1.npy".format(walking_style))
electricity_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                 "/{}/np_arrays/electricity_std_list_2.npy".format(walking_style))
electricity_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                 "/{}/np_arrays/electricity_std_list_3.npy".format(walking_style))
electricity_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                 "/{}/np_arrays/electricity_std_list_4.npy".format(walking_style))
final_x_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/final_x_list_0.npy".format(walking_style))
final_x_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/final_x_list_1.npy".format(walking_style))
final_x_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/final_x_list_2.npy".format(walking_style))
final_x_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/final_x_list_3.npy".format(walking_style))
final_x_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/final_x_list_4.npy".format(walking_style))
forward_return_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                "/{}/np_arrays/forward_return_list_0.npy".format(walking_style))
forward_return_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                "/{}/np_arrays/forward_return_list_1.npy".format(walking_style))
forward_return_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                "/{}/np_arrays/forward_return_list_2.npy".format(walking_style))
forward_return_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                "/{}/np_arrays/forward_return_list_3.npy".format(walking_style))
forward_return_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                "/{}/np_arrays/forward_return_list_4.npy".format(walking_style))
forward_return_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                    "/{}/np_arrays/forward_return_std_list_0.npy".format(walking_style))
forward_return_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                    "/{}/np_arrays/forward_return_std_list_1.npy".format(walking_style))
forward_return_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                    "/{}/np_arrays/forward_return_std_list_2.npy".format(walking_style))
forward_return_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                    "/{}/np_arrays/forward_return_std_list_3.npy".format(walking_style))
forward_return_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                                    "/{}/np_arrays/forward_return_std_list_4.npy".format(walking_style))
joint_lim_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/joint_lim_list_0.npy".format(walking_style))
joint_lim_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/joint_lim_list_1.npy".format(walking_style))
joint_lim_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/joint_lim_list_2.npy".format(walking_style))
joint_lim_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/joint_lim_list_3.npy".format(walking_style))
joint_lim_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/joint_lim_list_4.npy".format(walking_style))
joint_lim_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                               "/{}/np_arrays/joint_lim_std_list_0.npy".format(walking_style))
joint_lim_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                               "/{}/np_arrays/joint_lim_std_list_1.npy".format(walking_style))
joint_lim_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                               "/{}/np_arrays/joint_lim_std_list_2.npy".format(walking_style))
joint_lim_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                               "/{}/np_arrays/joint_lim_std_list_3.npy".format(walking_style))
joint_lim_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                               "/{}/np_arrays/joint_lim_std_list_4.npy".format(walking_style))
z_pos_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                       "/{}/np_arrays/z_pos_list_0.npy".format(walking_style))
z_pos_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                       "/{}/np_arrays/z_pos_list_1.npy".format(walking_style))
z_pos_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                       "/{}/np_arrays/z_pos_list_2.npy".format(walking_style))
z_pos_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                       "/{}/np_arrays/z_pos_list_3.npy".format(walking_style))
z_pos_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                       "/{}/np_arrays/z_pos_list_4.npy".format(walking_style))
z_pos_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/z_pos_std_list_0.npy".format(walking_style))
z_pos_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/z_pos_std_list_1.npy".format(walking_style))
z_pos_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/z_pos_std_list_2.npy".format(walking_style))
z_pos_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/z_pos_std_list_3.npy".format(walking_style))
z_pos_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                           "/{}/np_arrays/z_pos_std_list_4.npy".format(walking_style))
joint_0_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_0_list_0.npy".format(walking_style))
joint_0_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_0_list_1.npy".format(walking_style))
joint_0_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_0_list_2.npy".format(walking_style))
joint_0_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_0_list_3.npy".format(walking_style))
joint_0_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_0_list_4.npy".format(walking_style))
joint_0_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_0_std_list_0.npy".format(walking_style))
joint_0_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_0_std_list_1.npy".format(walking_style))
joint_0_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_0_std_list_2.npy".format(walking_style))
joint_0_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_0_std_list_3.npy".format(walking_style))
joint_0_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_0_std_list_4.npy".format(walking_style))
joint_1_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_1_list_0.npy".format(walking_style))
joint_1_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_1_list_1.npy".format(walking_style))
joint_1_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_1_list_2.npy".format(walking_style))
joint_1_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_1_list_3.npy".format(walking_style))
joint_1_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_1_list_4.npy".format(walking_style))
joint_1_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_1_std_list_0.npy".format(walking_style))
joint_1_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_1_std_list_1.npy".format(walking_style))
joint_1_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_1_std_list_2.npy".format(walking_style))
joint_1_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_1_std_list_3.npy".format(walking_style))
joint_1_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_1_std_list_4.npy".format(walking_style))
joint_2_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_2_list_0.npy".format(walking_style))
joint_2_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_2_list_1.npy".format(walking_style))
joint_2_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_2_list_2.npy".format(walking_style))
joint_2_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_2_list_3.npy".format(walking_style))
joint_2_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_2_list_4.npy".format(walking_style))
joint_2_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_2_std_list_0.npy".format(walking_style))
joint_2_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_2_std_list_1.npy".format(walking_style))
joint_2_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_2_std_list_2.npy".format(walking_style))
joint_2_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_2_std_list_3.npy".format(walking_style))
joint_2_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_2_std_list_4.npy".format(walking_style))
joint_3_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_3_list_0.npy".format(walking_style))
joint_3_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_3_list_1.npy".format(walking_style))
joint_3_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_3_list_2.npy".format(walking_style))
joint_3_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_3_list_3.npy".format(walking_style))
joint_3_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_3_list_4.npy".format(walking_style))
joint_3_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_3_std_list_0.npy".format(walking_style))
joint_3_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_3_std_list_1.npy".format(walking_style))
joint_3_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_3_std_list_2.npy".format(walking_style))
joint_3_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_3_std_list_3.npy".format(walking_style))
joint_3_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_3_std_list_4.npy".format(walking_style))
joint_4_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_4_list_0.npy".format(walking_style))
joint_4_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_4_list_1.npy".format(walking_style))
joint_4_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_4_list_2.npy".format(walking_style))
joint_4_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_4_list_3.npy".format(walking_style))
joint_4_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_4_list_4.npy".format(walking_style))
joint_4_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_4_std_list_0.npy".format(walking_style))
joint_4_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_4_std_list_1.npy".format(walking_style))
joint_4_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_4_std_list_2.npy".format(walking_style))
joint_4_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_4_std_list_3.npy".format(walking_style))
joint_4_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_4_std_list_4.npy".format(walking_style))
joint_5_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_5_list_0.npy".format(walking_style))
joint_5_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_5_list_1.npy".format(walking_style))
joint_5_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_5_list_2.npy".format(walking_style))
joint_5_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_5_list_3.npy".format(walking_style))
joint_5_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_5_list_4.npy".format(walking_style))
joint_5_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_5_std_list_0.npy".format(walking_style))
joint_5_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_5_std_list_1.npy".format(walking_style))
joint_5_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_5_std_list_2.npy".format(walking_style))
joint_5_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_5_std_list_3.npy".format(walking_style))
joint_5_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_5_std_list_4.npy".format(walking_style))
joint_6_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_6_list_0.npy".format(walking_style))
joint_6_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_6_list_1.npy".format(walking_style))
joint_6_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_6_list_2.npy".format(walking_style))
joint_6_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_6_list_3.npy".format(walking_style))
joint_6_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_6_list_4.npy".format(walking_style))
joint_6_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_6_std_list_0.npy".format(walking_style))
joint_6_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_6_std_list_1.npy".format(walking_style))
joint_6_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_6_std_list_2.npy".format(walking_style))
joint_6_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_6_std_list_3.npy".format(walking_style))
joint_6_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_6_std_list_4.npy".format(walking_style))
joint_7_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_7_list_0.npy".format(walking_style))
joint_7_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_7_list_1.npy".format(walking_style))
joint_7_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_7_list_2.npy".format(walking_style))
joint_7_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_7_list_3.npy".format(walking_style))
joint_7_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                         "/{}/np_arrays/joint_7_list_4.npy".format(walking_style))
joint_7_std_list_0 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_7_std_list_0.npy".format(walking_style))
joint_7_std_list_1 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_7_std_list_1.npy".format(walking_style))
joint_7_std_list_2 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_7_std_list_2.npy".format(walking_style))
joint_7_std_list_3 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_7_std_list_3.npy".format(walking_style))
joint_7_std_list_4 = np.load("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
                             "/{}/np_arrays/joint_7_std_list_4.npy".format(walking_style))

# means aus den arrays von oben berechnen
skia = 0
electricity_mean = np.mean(np.array([electricity_list_0, electricity_list_1, electricity_list_2,
                                     electricity_list_3, electricity_list_4]), axis=0)
electricity_std_mean = np.mean(np.array([electricity_std_list_0, electricity_std_list_1, electricity_std_list_2,
                                         electricity_std_list_3, electricity_std_list_4]), axis=0)
final_x_mean = np.mean(np.array([final_x_list_0, final_x_list_1, final_x_list_2,
                                 final_x_list_3, final_x_list_4]), axis=0)
forward_return_mean = np.mean(np.array([forward_return_list_0, forward_return_list_1, forward_return_list_2,
                                        forward_return_list_3, forward_return_list_4]), axis=0)
forward_return_std_mean = np.mean(np.array([forward_return_std_list_0, forward_return_std_list_1,
                                            forward_return_std_list_2, forward_return_std_list_3,
                                            forward_return_std_list_4]), axis=0)
joint_lim_mean = np.mean(np.array([joint_lim_list_0, joint_lim_list_1, joint_lim_list_2,
                                   joint_lim_list_3, joint_lim_list_4]), axis=0)
joint_lim_std_mean = np.mean(np.array([joint_lim_std_list_0, joint_lim_std_list_1,
                                       joint_lim_std_list_2, joint_lim_std_list_3,
                                       joint_lim_std_list_4]), axis=0)
z_pos_mean = np.mean(np.array([z_pos_list_0, z_pos_list_1, z_pos_list_2, z_pos_list_3,
                               z_pos_list_4]), axis=0)
z_pos_std_mean = np.mean(np.array([z_pos_std_list_0, z_pos_std_list_1, z_pos_std_list_2,
                                   z_pos_std_list_3, z_pos_std_list_4]), axis=0)
joint_0_mean = np.mean(np.array([joint_0_list_0, joint_0_list_1, joint_0_list_2,
                                 joint_0_list_3, joint_0_list_4]), axis=0)
joint_0_std_mean = np.mean(np.array([joint_0_std_list_0, joint_0_std_list_1, joint_0_std_list_2,
                                     joint_0_std_list_3, joint_0_std_list_4]), axis=0)
joint_1_mean = np.mean(np.array([joint_1_list_0, joint_1_list_1, joint_1_list_2,
                                 joint_1_list_3, joint_1_list_4]), axis=0)
joint_1_std_mean = np.mean(np.array([joint_1_std_list_0, joint_1_std_list_1, joint_1_std_list_2,
                                     joint_1_std_list_3, joint_1_std_list_4]), axis=0)
joint_2_mean = np.mean(np.array([joint_2_list_0, joint_2_list_1, joint_2_list_2,
                                 joint_2_list_3, joint_2_list_4]), axis=0)
joint_2_std_mean = np.mean(np.array([joint_2_std_list_0, joint_2_std_list_1, joint_2_std_list_2,
                                     joint_2_std_list_3, joint_2_std_list_4]), axis=0)
joint_3_mean = np.mean(np.array([joint_3_list_0, joint_3_list_1, joint_3_list_2,
                                 joint_3_list_3, joint_3_list_4]), axis=0)
joint_3_std_mean = np.mean(np.array([joint_3_std_list_0, joint_3_std_list_1, joint_3_std_list_2,
                                     joint_3_std_list_3, joint_3_std_list_4]), axis=0)
joint_4_mean = np.mean(np.array([joint_4_list_0, joint_4_list_1, joint_4_list_2,
                                 joint_4_list_3, joint_4_list_4]), axis=0)
joint_4_std_mean = np.mean(np.array([joint_4_std_list_0, joint_4_std_list_1, joint_4_std_list_2,
                                     joint_4_std_list_3, joint_4_std_list_4]), axis=0)
joint_5_mean = np.mean(np.array([joint_5_list_0, joint_5_list_1, joint_5_list_2,
                                 joint_5_list_3, joint_5_list_4]), axis=0)
joint_5_std_mean = np.mean(np.array([joint_5_std_list_0, joint_5_std_list_1, joint_5_std_list_2,
                                     joint_5_std_list_3, joint_5_std_list_4]), axis=0)
joint_6_mean = np.mean(np.array([joint_6_list_0, joint_6_list_1, joint_6_list_2,
                                 joint_6_list_3, joint_6_list_4]), axis=0)
joint_6_std_mean = np.mean(np.array([joint_6_std_list_0, joint_6_std_list_1, joint_6_std_list_2,
                                     joint_6_std_list_3, joint_6_std_list_4]), axis=0)
joint_7_mean = np.mean(np.array([joint_7_list_0, joint_7_list_1, joint_7_list_2,
                                 joint_7_list_3, joint_7_list_4]), axis=0)
joint_7_std_mean = np.mean(np.array([joint_7_std_list_0, joint_7_std_list_1, joint_7_std_list_2,
                                     joint_7_std_list_3, joint_7_std_list_4]), axis=0)

# a Walking Style Plots -- fejz 0 std; joints 0.5 std
fig3, ax3 = plt.subplots()
f_tmp = np.array(forward_return_mean)
f_tmp_std = np.array(forward_return_std_mean)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(forward_return_mean)), forward_return_mean, color="#187AB2", label="forward")
ax3.fill_between(range(len(forward_return_mean)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor="#187AB2")
f_tmp = np.array(electricity_mean)
f_tmp_std = np.array(electricity_std_mean)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(electricity_mean)), electricity_mean, color='#D32826', label="electricity")
ax3.fill_between(range(len(electricity_mean)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#D32826')
f_tmp = np.array(joint_lim_mean)
f_tmp_std = np.array(joint_lim_std_mean)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(joint_lim_mean)), joint_lim_mean, color='#FC800B', label="joint_limit")
ax3.fill_between(range(len(joint_lim_mean)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#FC800B')
f_tmp = np.array(z_pos_mean)
f_tmp_std = np.array(z_pos_std_mean)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(z_pos_mean)), z_pos_mean, color='#309F2D', label="z_pos")
ax3.fill_between(range(len(z_pos_mean)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#309F2D')
plt.yticks(np.arange(-1, 2.5, step=0.5))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/a_fejz.pdf".format(walking_style), bbox_inches='tight')

fig4, ax4 = plt.subplots()
j = np.array(joint_0_mean)
j_std = np.array(joint_0_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_0_mean)), joint_0_mean, color="#187AB2", label="j_0")
ax4.fill_between(range(len(joint_0_mean)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax4.plot(range(len(joint_1_list_n)), joint_1_list_n, color='g', label="j_1")
j = np.array(joint_2_mean)
j_std = np.array(joint_2_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_2_mean)), joint_2_mean, color='#D32826', label="j_2")
ax4.fill_between(range(len(joint_2_mean)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax4.plot(range(len(joint_3_list_n)), joint_3_list_n, color='c', label="j_3")
j = np.array(joint_4_mean)
j_std = np.array(joint_4_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_4_mean)), joint_4_mean, color='#FC800B', label="j_4")
ax4.fill_between(range(len(joint_4_mean)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax4.plot(range(len(joint_5_list_n)), joint_5_list_n, color='y', label="j_5")
j = np.array(joint_6_mean)
j_std = np.array(joint_6_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_6_mean)), joint_6_mean, color='#309F2D', label="j_6")
ax4.fill_between(range(len(joint_6_mean)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
# ax4.plot(range(len(joint_7_list_n)), joint_7_list_n, color='grey', label="j_7")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/a_joints_gerade_half.pdf".format(walking_style), bbox_inches='tight')

fig5, ax5 = plt.subplots()
# ax5.plot(range(len(joint_0_list_n)), joint_0_list_n, color='b', label="j_0")
j = np.array(joint_1_mean)
j_std = np.array(joint_1_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_1_mean)), joint_1_mean, color="#187AB2", label="j_1")
ax5.fill_between(range(len(joint_1_mean)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax5.plot(range(len(joint_2_list_n)), joint_2_list_n, color='r', label="j_2")
j = np.array(joint_3_mean)
j_std = np.array(joint_3_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_3_mean)), joint_3_mean, color='#D32826', label="j_3")
ax5.fill_between(range(len(joint_3_mean)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax5.plot(range(len(joint_4_list_n)), joint_4_list_n, color='m', label="j_4")
j = np.array(joint_5_mean)
j_std = np.array(joint_5_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_5_mean)), joint_5_mean, color='#FC800B', label="j_5")
ax5.fill_between(range(len(joint_5_mean)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax5.plot(range(len(joint_6_list_n)), joint_6_list_n, color='lime', label="j_6")
j = np.array(joint_7_mean)
j_std = np.array(joint_7_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_7_mean)), joint_7_mean, color='#309F2D', label="j_7")
ax5.fill_between(range(len(joint_7_mean)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/a_joints_ungerade_half.pdf".format(walking_style), bbox_inches='tight')

# b walking style plots -- joints 0.5 std
fig6, ax6 = plt.subplots()
j = np.array(joint_0_mean)
j_std = np.array(0.5*joint_0_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_0_mean)), joint_0_mean, color="#187AB2", label="j_0")
ax6.fill_between(range(len(joint_0_mean)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax4.plot(range(len(joint_1_list_n)), joint_1_list_n, color='g', label="j_1")
j = np.array(joint_2_mean)
j_std = np.array(0.5*joint_2_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_2_mean)), joint_2_mean, color='#D32826', label="j_2")
ax6.fill_between(range(len(joint_2_mean)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax4.plot(range(len(joint_3_list_n)), joint_3_list_n, color='c', label="j_3")
j = np.array(joint_4_mean)
j_std = np.array(0.5*joint_4_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_4_mean)), joint_4_mean, color='#FC800B', label="j_4")
ax6.fill_between(range(len(joint_4_mean)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax4.plot(range(len(joint_5_list_n)), joint_5_list_n, color='y', label="j_5")
j = np.array(joint_6_mean)
j_std = np.array(0.5*joint_6_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_6_mean)), joint_6_mean, color='#309F2D', label="j_6")
ax6.fill_between(range(len(joint_6_mean)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
# ax4.plot(range(len(joint_7_list_n)), joint_7_list_n, color='grey', label="j_7")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/b_joints_gerade_quater.pdf".format(walking_style), bbox_inches='tight')

fig7, ax7 = plt.subplots()
# ax5.plot(range(len(joint_0_list_n)), joint_0_list_n, color='b', label="j_0")
j = np.array(joint_1_mean)
j_std = np.array(0.5*joint_1_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_1_mean)), joint_1_mean, color="#187AB2", label="j_1")
ax7.fill_between(range(len(joint_1_mean)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax5.plot(range(len(joint_2_list_n)), joint_2_list_n, color='r', label="j_2")
j = np.array(joint_3_mean)
j_std = np.array(0.5*joint_3_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_3_mean)), joint_3_mean, color='#D32826', label="j_3")
ax7.fill_between(range(len(joint_3_mean)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax5.plot(range(len(joint_4_list_n)), joint_4_list_n, color='m', label="j_4")
j = np.array(joint_5_mean)
j_std = np.array(0.5*joint_5_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_5_mean)), joint_5_mean, color='#FC800B', label="j_5")
ax7.fill_between(range(len(joint_5_mean)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax5.plot(range(len(joint_6_list_n)), joint_6_list_n, color='lime', label="j_6")
j = np.array(joint_7_mean)
j_std = np.array(0.5*joint_7_std_mean)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_7_mean)), joint_7_mean, color='#309F2D', label="j_7")
ax7.fill_between(range(len(joint_7_mean)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/b_joints_ungerade_quater.pdf".format(walking_style), bbox_inches='tight')

# c Walking Style Plots -- no std
fig8, ax8 = plt.subplots()
ax8.plot(range(len(forward_return_mean)), forward_return_mean, color="#187AB2", label="forward")
ax8.plot(range(len(electricity_mean)), electricity_mean, color='#D32826', label="electricity")
ax8.plot(range(len(joint_lim_mean)), joint_lim_mean, color='#FC800B', label="joint_limit")
ax8.plot(range(len(z_pos_mean)), z_pos_mean, color='#309F2D', label="z_pos")
plt.yticks(np.arange(-1, 2.5, step=0.5))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/c_fejz_no_std.pdf".format(walking_style), bbox_inches='tight')

fig9, ax9 = plt.subplots()
ax9.plot(range(len(joint_0_mean)), joint_0_mean, color="#187AB2", label="j_0")
ax9.plot(range(len(joint_2_mean)), joint_2_mean, color='#D32826', label="j_2")
ax9.plot(range(len(joint_4_mean)), joint_4_mean, color='#FC800B', label="j_4")
ax9.plot(range(len(joint_6_mean)), joint_6_mean, color='#309F2D', label="j_6")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/c_joints_gerade_no_std.pdf".format(walking_style), bbox_inches='tight')

fig10, ax10 = plt.subplots()
ax10.plot(range(len(joint_1_mean)), joint_1_mean, color="#187AB2", label="j_1")
ax10.plot(range(len(joint_3_mean)), joint_3_mean, color='#D32826', label="j_3")
ax10.plot(range(len(joint_5_mean)), joint_5_mean, color='#FC800B', label="j_5")
ax10.plot(range(len(joint_7_mean)), joint_7_mean, color='#309F2D', label="j_7")
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/results/c_joints_ungerade_no_std.pdf".format(walking_style), bbox_inches='tight')

