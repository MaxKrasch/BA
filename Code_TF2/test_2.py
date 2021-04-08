import numpy as np
import gym
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_networks import ActorNN, CriticNN
import tensorflow as tf
from replay_buffer import ReplayBuffer
from noise import NormalActionNoise
import os
import pybulletgym
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})


reward_fcn_name = "pb_normal_prove_3"


def update_network_parameters(q1, q1_target, q2, q2_target, mu, mu_target, tau):

    # update target_critic weights
    tmp_weights = []
    q_targets_weights = q1_target.weights
    for i, weight in enumerate(q1.weights):
        tmp_weights.append(tau * weight + (1 - tau) * q_targets_weights[i])
    q1_target.set_weights(tmp_weights)

    tmp_weights = []
    q_targets_weights = q2_target.weights
    for i, weight in enumerate(q2.weights):
        tmp_weights.append(tau * weight + (1 - tau) * q_targets_weights[i])
    q2_target.set_weights(tmp_weights)

    # update target_actor weights
    tmp_weights = []
    mu_target_weights = mu_target.weights
    for i, weight in enumerate(mu.weights):
        tmp_weights.append(tau * weight + (1 - tau) * mu_target_weights[i])
    mu_target.set_weights(tmp_weights)

    return q1_target, q2_target, mu_target


def ddpg(episode, breaking_step, reward_name):
    env = gym.make('AntPyBulletEnv-v0')
    cumulus_steps = 0
    episode_steps = 0

    # randomly initialize critics and actor with weights and biases
    q1 = CriticNN()
    q1.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    q2 = CriticNN()
    q2.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    mu = ActorNN(env.action_space.shape[0])
    mu.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    # initialize target networks
    q1_target = CriticNN()
    q1_target.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    q2_target = CriticNN()
    q2_target.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    mu_target = ActorNN(env.action_space.shape[0])
    mu_target.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    q1_target, q2_target, mu_target = update_network_parameters(q1, q1_target, q2, q2_target, mu, mu_target, 0.005)

    # initialize replay buffer (actor critic train only after batch is full 64!)
    replay_buffer = ReplayBuffer(1000000, env.observation_space.shape[0], env.action_space.shape[0])

    performance = []
    avg_return = []
    time_step_reward = []
    avg_time_step_reward = []
    a_c = 0
    b_c = 0
    c_c = 0
    d_c = 0
    e_c = 0
    f_c = 0
    for e in range(episode):

        # receive initial observation state s1 (observation = s1)
        # env.render()
        observation = env.reset()
        state = tf.convert_to_tensor([observation], dtype=tf.float32)

        max_steps = 1000
        min_action = env.action_space.low[0]
        max_action = env.action_space.high[0]
        update_frequency = 2
        learn_count = 0
        score = 0
        for i in range(max_steps):

            # select an action a_t = mu(state) + noise
            noise = NormalActionNoise(0, 0.1)
            action = mu(state) + np.random.normal(noise.mean, noise.sigma)
            proto_tensor = tf.make_tensor_proto(action)
            action = tf.make_ndarray(proto_tensor)
            action = action[0]

            # execute action a_t and observe reward, and next state
            next_state, reward, done, _ = env.step(action)

            # store transition in replay buffer
            replay_buffer.store_transition(state, action, reward, next_state, done)

            # if there are enough transitions in the replay buffer
            batch_size = 100
            if replay_buffer.mem_cntr >= batch_size:

                # sample a random mini batch of n=64 transitions
                buff_state, buff_action, buff_reward, buff_next_state, buff_done = replay_buffer.sample_buffer(batch_size)

                states = tf.convert_to_tensor(buff_state, dtype=tf.float32)
                next_states = tf.convert_to_tensor(buff_next_state, dtype=tf.float32)
                rewards = tf.convert_to_tensor(buff_reward, dtype=tf.float32)
                actions = tf.convert_to_tensor(buff_action, dtype=tf.float32)

                # train critics
                with tf.GradientTape(persistent=True) as tape:

                    # calculate which actions target_actor chooses and add noise
                    target_actions = mu_target(next_states) + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)
                    target_actions = tf.clip_by_value(target_actions, min_action, max_action)

                    # calculate next_q_values of the critic by feeding the next state and from actor chosen actions
                    next_critic_value1 = tf.squeeze(q1_target(next_states, target_actions), 1)
                    next_critic_value2 = tf.squeeze(q2_target(next_states, target_actions), 1)

                    # calculate q values of critic actual state
                    critic_value1 = tf.squeeze(q1(states, actions), 1)
                    critic_value2 = tf.squeeze(q2(states, actions), 1)

                    # use smaller q value from the 2 critics
                    next_critic_value = tf.math.minimum(next_critic_value1, next_critic_value2)

                    # calculate target values: yt = rt + gamma * q_target(s_t+1, mu_target(s_t+1)); with t = time step
                    y = rewards + 0.99 * next_critic_value * (1 - buff_done)

                    # calculate the loss between critic and target_critic
                    critic1_loss = keras.losses.MSE(y, critic_value1)
                    critic2_loss = keras.losses.MSE(y, critic_value2)

                # update critics by minimized the loss (critic_loss) and using Adam optimizer
                critic1_network_gradient = tape.gradient(critic1_loss, q1.trainable_variables)
                critic2_network_gradient = tape.gradient(critic2_loss, q2.trainable_variables)
                q1.optimizer.apply_gradients(zip(critic1_network_gradient, q1.trainable_variables))
                q2.optimizer.apply_gradients(zip(critic2_network_gradient, q2.trainable_variables))

                learn_count += 1

                # train actor
                if learn_count % update_frequency == 0:
                    with tf.GradientTape() as tape:
                        new_policy_actions = mu(states)
                        # check if - or + (descent or ascent) not sure yet
                        actor_loss = -q1(states, new_policy_actions)
                        actor_loss = tf.math.reduce_mean(actor_loss)

                    # update the actor policy using the sampled policy gradient
                    actor_network_gradient = tape.gradient(actor_loss, mu.trainable_variables)
                    mu.optimizer.apply_gradients(zip(actor_network_gradient, mu.trainable_variables))

                    # update the target networks
                    update_network_parameters(q1, q1_target, q2, q2_target, mu, mu_target, 0.005)

            time_step_reward.append(reward)
            avg_time_step_reward_short = np.mean(time_step_reward[-50:])
            avg_time_step_reward.append(avg_time_step_reward_short)
            if done:
                performance.append(score)
                avg_reward = np.mean(performance[-50:])
                avg_return.append(avg_reward)
                cumulus_steps += i
                print("episode: {}/{}, score: {}, avg_score: {}, ep_steps: {}, cumulus_steps: {}"
                      .format(e, episode, score, avg_reward, i, cumulus_steps))

                if 10000 < cumulus_steps < 11000 and a_c == 0:
                    a_c = 1
                    if not os.path.exists("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                        os.mkdir("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                    mu.save_weights("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)

                if 150000 < cumulus_steps < 151000 and b_c == 0:
                    b_c = 1
                    if not os.path.exists("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                        os.mkdir("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                    mu.save_weights("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)

                # if 350000 < cumulus_steps < 351000 and c_c == 0:
                #     c_c = 1
                #     if not os.path.exists("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                #         os.mkdir("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                #     mu.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                #     np.save("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                #     np.save("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                #     np.save("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                #     np.save("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)

                if 550000 < cumulus_steps < 551000 and d_c == 0:
                    d_c = 1
                    if not os.path.exists("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                        os.mkdir("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                    mu.save_weights("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)

                if 750000 < cumulus_steps < 751000 and e_c == 0:
                    e_c = 1
                    if not os.path.exists("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                        os.mkdir("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                    mu.save_weights("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)

                if 1000000 < cumulus_steps < 1001000 and f_c == 0:
                    f_c = 1
                    mu.save_weights("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_return{}".format(reward_name, cumulus_steps), avg_return)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/time_step_reward{}".format(reward_name, cumulus_steps), time_step_reward)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/performance{}".format(reward_name, cumulus_steps), performance)
                    np.save("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/avg_time_step_reward{}".format(reward_name, cumulus_steps), avg_time_step_reward)
                break

            score += reward
            state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        # stop learning after certain time steps
        if cumulus_steps > breaking_step:
            break

    return avg_return, mu, performance, time_step_reward, avg_time_step_reward


def test(mu_render, e, train_bool, weight_string):
    env = gym.make('AntPyBulletEnv-v0')
    if not train_bool:
        mu_render.load_weights(weight_string)
    final_x_list = []
    forward_return_list = []
    forward_return_std_list = []
    electricity_list = []
    electricity_std_list = []
    joint_lim_list = []
    joint_lim_std_list = []
    z_pos_list = []
    z_pos_std_list = []
    cumulus_steps_list = []
    joint_0_list = []
    joint_0_std_list = []
    joint_1_list = []
    joint_1_std_list = []
    joint_2_list = []
    joint_2_std_list = []
    joint_3_list = []
    joint_3_std_list = []
    joint_4_list = []
    joint_4_std_list = []
    joint_5_list = []
    joint_5_std_list = []
    joint_6_list = []
    joint_6_std_list = []
    joint_7_list = []
    joint_7_std_list = []
    std_time = 0.5
    for i in range(e):
        print(i)
        done = 0
        ep_reward = 0
        step = 0
        # env.render()
        observation = env.reset()
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        final_x_pos = 0
        episode_alive = 0
        forward_ep_list = []
        electricity_ep_list = []
        jointlim_ep_list = []
        z_ep_list = []
        joint_0_ep_list = []
        joint_1_ep_list = []
        joint_2_ep_list = []
        joint_3_ep_list = []
        joint_4_ep_list = []
        joint_5_ep_list = []
        joint_6_ep_list = []
        joint_7_ep_list = []
        while not done:
            action = mu_render(state)
            proto_tensor = tf.make_tensor_proto(action)
            action = tf.make_ndarray(proto_tensor)
            action = action[0]
            # action[4] = 0
            # action[5] = 0
            # print(action)
            next_state, reward, done, _ = env.step(action)
            # print(next_state[24], next_state[27], next_state[25], next_state[26])
            reward_list = env.env.rewards
            reward = reward_list[1]
            # print(next_state[6], next_state[7])
            z_pos = env.env.robot.body_xyz[2]
            state = tf.convert_to_tensor([next_state], dtype=tf.float32)
            ep_reward += reward
            final_x_pos = env.env.robot.body_xyz[0]
            episode_alive += reward_list[0]
            forward_ep_list.append(reward)
            electricity_ep_list.append(reward_list[2])
            jointlim_ep_list.append(reward_list[3])
            z_ep_list.append(z_pos)
            # print(next_state[8],next_state[10],next_state[12],next_state[14],next_state[16],next_state[18],next_state[20],next_state[22])
            joint_0_ep_list.append(next_state[8] * 45)
            joint_1_ep_list.append((next_state[10] + 2) * 35)
            joint_2_ep_list.append(next_state[12] * 45)
            joint_3_ep_list.append((next_state[14] - 2) * (-35))
            joint_4_ep_list.append(next_state[16] * 45)
            joint_5_ep_list.append((next_state[18] - 2) * (-35))
            joint_6_ep_list.append(next_state[20] * 45)
            joint_7_ep_list.append((next_state[22] + 2) * 35)
            step += 1
        print("Final x position: {}".format(final_x_pos))
        final_x_list.append(final_x_pos)
        print("Episode forward return: {}".format(np.mean(forward_ep_list)))
        forward_return_list.append(np.mean(forward_ep_list))
        forward_return_std_list.append(np.std(forward_ep_list))
        print("Average Electricity costs: {}".format(np.mean(electricity_ep_list)))
        electricity_list.append(np.mean(electricity_ep_list))
        electricity_std_list.append(np.std(electricity_ep_list))
        print("Joints at limits costs: {}".format(np.mean(jointlim_ep_list)))
        joint_lim_list.append(np.mean(jointlim_ep_list))
        joint_lim_std_list.append(np.std(jointlim_ep_list))
        print("Average z position: {}".format(np.mean(z_ep_list)))
        z_pos_list.append(np.mean(z_ep_list))
        z_pos_std_list.append(np.std(z_ep_list))
        print("cumulus steps: {}".format(step))
        cumulus_steps_list.append(step)
        # joints
        joint_0_list.append(np.mean(joint_0_ep_list))
        joint_0_std_list.append(std_time*np.std(joint_0_ep_list))
        joint_1_list.append(np.mean(joint_1_ep_list))
        joint_1_std_list.append(std_time*np.std(joint_1_ep_list))
        joint_2_list.append(np.mean(joint_2_ep_list))
        joint_2_std_list.append(std_time*np.std(joint_2_ep_list))
        joint_3_list.append(np.mean(joint_3_ep_list))
        joint_3_std_list.append(std_time*np.std(joint_3_ep_list))
        joint_4_list.append(np.mean(joint_4_ep_list))
        joint_4_std_list.append(std_time*np.std(joint_4_ep_list))
        joint_5_list.append(np.mean(joint_5_ep_list))
        joint_5_std_list.append(std_time*np.std(joint_5_ep_list))
        joint_6_list.append(np.mean(joint_6_ep_list))
        joint_6_std_list.append(std_time*np.std(joint_6_ep_list))
        joint_7_list.append(np.mean(joint_7_ep_list))
        joint_7_std_list.append(std_time*np.std(joint_7_ep_list))
    print("Mean final x position: {}".format(np.mean(final_x_list)))
    print("Mean episode forward return: {}".format(np.mean(forward_return_list)))
    print("Mean episode electricity costs: {}".format(np.mean(electricity_list)))
    print("Mean joint limit costs: {}".format(np.mean(joint_lim_list)))
    print("Mean episodes mean z pos: {}".format(np.mean(z_pos_list)))
    print("Mean cumulus steps: {}".format(np.mean(cumulus_steps_list)))
    return final_x_list, forward_return_list, forward_return_std_list, electricity_list, electricity_std_list, \
        joint_lim_list, joint_lim_std_list, z_pos_list, z_pos_std_list, cumulus_steps_list, \
        joint_0_list, joint_0_std_list, joint_1_list, joint_1_std_list, joint_2_list, joint_2_std_list, \
        joint_3_list, joint_3_std_list, joint_4_list, joint_4_std_list, joint_5_list, joint_5_std_list, \
        joint_6_list, joint_6_std_list, joint_7_list, joint_7_std_list


# main starts
train = False
break_step = 1002000
agent_weights = "none"

walking_type = "tight_walk"
model_name = "pb_tight_prove_0"
mu_rendering = "mu1000768.h5"
model_count = "_0"

if not train:
    break_step = 100
    agent_weights = "/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves" \
                    "/{}/{}/{}".format(walking_type, model_name, mu_rendering)

episodes = 500000
overall_performance, mu, per, time_step_rew, avg_time_step_rew = ddpg(episodes, break_step, reward_fcn_name)

# plot performance
if train:
    fig, ax = plt.subplots()
    ax.plot(range(len(overall_performance)), overall_performance, color='r')
    ax.plot(range(len(per)), per, 'b')
    plt.xlabel("Episodes")
    plt.ylabel("Performance")
    plt.grid(True)
    # plt.show()
    plt.savefig("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/figure1.pdf".format(reward_fcn_name),
                bbox_inches='tight')
    fig2, ax2 = plt.subplots()
    ax2.plot(range(len(time_step_rew)), time_step_rew, color='y')
    ax2.plot(range(len(avg_time_step_rew)), avg_time_step_rew, 'b')
    plt.xlabel("Time Steps")
    plt.ylabel("Performance")
    plt.grid(True)
    # plt.show()
    plt.savefig("/var/tmp/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}/figure2.pdf".format(reward_fcn_name),
                bbox_inches='tight')

# test and render
eps = 100
final_x_list1, forward_return_list1, forward_return_std_list1, electricity_list1, electricity_std_list1, \
  joint_lim_list1, joint_lim_std_list1, z_pos_list1, z_pos_std_list1, cumulus_steps_list1, \
  joint_0_list_n, joint_0_std_list_n, joint_1_list_n, joint_1_std_list_n, joint_2_list_n, joint_2_std_list_n, \
  joint_3_list_n, joint_3_std_list_n, joint_4_list_n, joint_4_std_list_n, joint_5_list_n, joint_5_std_list_n, \
  joint_6_list_n, joint_6_std_list_n, \
  joint_7_list_n, joint_7_std_list_n = test(mu, eps, train, agent_weights)

final_x_list1 = np.array(final_x_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/final_x_list{}".format(walking_type, model_count), final_x_list1)
forward_return_list1 = np.array(forward_return_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/forward_return_list{}".format(walking_type, model_count), forward_return_list1)
forward_return_std_list1 = np.array(forward_return_std_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/forward_return_std_list{}".format(walking_type, model_count), forward_return_std_list1)
electricity_list1 = np.array(electricity_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/electricity_list{}".format(walking_type, model_count), electricity_list1)
electricity_std_list1 = np.array(electricity_std_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/electricity_std_list{}".format(walking_type, model_count), electricity_std_list1)
joint_lim_list1 = np.array(joint_lim_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_lim_list{}".format(walking_type, model_count), joint_lim_list1)
joint_lim_std_list1 = np.array(joint_lim_std_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_lim_std_list{}".format(walking_type, model_count), joint_lim_std_list1)
z_pos_list1 = np.array(z_pos_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/z_pos_list{}".format(walking_type, model_count), z_pos_list1)
z_pos_std_list1 = np.array(z_pos_std_list1)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/z_pos_std_list{}".format(walking_type, model_count), z_pos_std_list1)
joint_0_list_n = np.array(joint_0_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_0_list{}".format(walking_type, model_count), joint_0_list_n)
joint_0_std_list_n = np.array(joint_0_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_0_std_list{}".format(walking_type, model_count), joint_0_std_list_n)
joint_1_list_n = np.array(joint_1_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_1_list{}".format(walking_type, model_count), joint_1_list_n)
joint_1_std_list_n = np.array(joint_1_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_1_std_list{}".format(walking_type, model_count), joint_1_std_list_n)
joint_2_list_n = np.array(joint_2_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_2_list{}".format(walking_type, model_count), joint_2_list_n)
joint_2_std_list_n = np.array(joint_2_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_2_std_list{}".format(walking_type, model_count), joint_2_std_list_n)
joint_3_list_n = np.array(joint_3_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_3_list{}".format(walking_type, model_count), joint_3_list_n)
joint_3_std_list_n = np.array(joint_3_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_3_std_list{}".format(walking_type, model_count), joint_3_std_list_n)
joint_4_list_n = np.array(joint_4_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_4_list{}".format(walking_type, model_count), joint_4_list_n)
joint_4_std_list_n = np.array(joint_4_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_4_std_list{}".format(walking_type, model_count), joint_4_std_list_n)
joint_5_list_n = np.array(joint_5_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_5_list{}".format(walking_type, model_count), joint_5_list_n)
joint_5_std_list_n = np.array(joint_5_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_5_std_list{}".format(walking_type, model_count), joint_5_std_list_n)
joint_6_list_n = np.array(joint_6_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_6_list{}".format(walking_type, model_count), joint_6_list_n)
joint_6_std_list_n = np.array(joint_6_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_6_std_list{}".format(walking_type, model_count), joint_6_std_list_n)
joint_7_list_n = np.array(joint_7_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_7_list{}".format(walking_type, model_count), joint_7_list_n)
joint_7_std_list_n = np.array(joint_7_std_list_n)
np.save("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models"
        "/proves/{}/np_arrays/joint_7_std_list{}".format(walking_type, model_count), joint_7_std_list_n)

# a Walking Style Plots -- fejz 0 std; joints 0.5 std
fig3, ax3 = plt.subplots()
f_tmp = np.array(forward_return_list1)
f_tmp_std = np.array(forward_return_std_list1)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(forward_return_list1)), forward_return_list1, color="#187AB2", label="forward")
ax3.fill_between(range(len(forward_return_list1)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor="#187AB2")
f_tmp = np.array(electricity_list1)
f_tmp_std = np.array(electricity_std_list1)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(electricity_list1)), electricity_list1, color='#D32826', label="electricity")
ax3.fill_between(range(len(electricity_list1)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#D32826')
f_tmp = np.array(joint_lim_list1)
f_tmp_std = np.array(joint_lim_std_list1)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(joint_lim_list1)), joint_lim_list1, color='#FC800B', label="joint_limit")
ax3.fill_between(range(len(joint_lim_list1)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#FC800B')
f_tmp = np.array(z_pos_list1)
f_tmp_std = np.array(z_pos_std_list1)
f_tmp_plus = f_tmp+f_tmp_std
f_tmp_minus = f_tmp-f_tmp_std
f_tmp_plus = list(f_tmp_plus)
f_tmp_minus = list(f_tmp_minus)
ax3.plot(range(len(z_pos_list1)), z_pos_list1, color='#309F2D', label="z_pos")
ax3.fill_between(range(len(z_pos_list1)), f_tmp_plus, f_tmp_minus, alpha=0.5, facecolor='#309F2D')
plt.yticks(np.arange(-1, 2.5, step=0.5))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/a_fejz.pdf".format(walking_type, model_name), bbox_inches='tight')

fig4, ax4 = plt.subplots()
j = np.array(joint_0_list_n)
j_std = np.array(joint_0_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_0_list_n)), joint_0_list_n, color="#187AB2", label="j_0")
ax4.fill_between(range(len(joint_0_list_n)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax4.plot(range(len(joint_1_list_n)), joint_1_list_n, color='g', label="j_1")
j = np.array(joint_2_list_n)
j_std = np.array(joint_2_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_2_list_n)), joint_2_list_n, color='#D32826', label="j_2")
ax4.fill_between(range(len(joint_2_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax4.plot(range(len(joint_3_list_n)), joint_3_list_n, color='c', label="j_3")
j = np.array(joint_4_list_n)
j_std = np.array(joint_4_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_4_list_n)), joint_4_list_n, color='#FC800B', label="j_4")
ax4.fill_between(range(len(joint_4_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax4.plot(range(len(joint_5_list_n)), joint_5_list_n, color='y', label="j_5")
j = np.array(joint_6_list_n)
j_std = np.array(joint_6_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax4.plot(range(len(joint_6_list_n)), joint_6_list_n, color='#309F2D', label="j_6")
ax4.fill_between(range(len(joint_6_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
# ax4.plot(range(len(joint_7_list_n)), joint_7_list_n, color='grey', label="j_7")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/a_joints_gerade_half.pdf".format(walking_type, model_name), bbox_inches='tight')

fig5, ax5 = plt.subplots()
# ax5.plot(range(len(joint_0_list_n)), joint_0_list_n, color='b', label="j_0")
j = np.array(joint_1_list_n)
j_std = np.array(joint_1_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_1_list_n)), joint_1_list_n, color="#187AB2", label="j_1")
ax5.fill_between(range(len(joint_1_list_n)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax5.plot(range(len(joint_2_list_n)), joint_2_list_n, color='r', label="j_2")
j = np.array(joint_3_list_n)
j_std = np.array(joint_3_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_3_list_n)), joint_3_list_n, color='#D32826', label="j_3")
ax5.fill_between(range(len(joint_3_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax5.plot(range(len(joint_4_list_n)), joint_4_list_n, color='m', label="j_4")
j = np.array(joint_5_list_n)
j_std = np.array(joint_5_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_5_list_n)), joint_5_list_n, color='#FC800B', label="j_5")
ax5.fill_between(range(len(joint_5_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax5.plot(range(len(joint_6_list_n)), joint_6_list_n, color='lime', label="j_6")
j = np.array(joint_7_list_n)
j_std = np.array(joint_7_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax5.plot(range(len(joint_7_list_n)), joint_7_list_n, color='#309F2D', label="j_7")
ax5.fill_between(range(len(joint_7_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/a_joints_ungerade_half.pdf".format(walking_type, model_name), bbox_inches='tight')

# b walking style plots -- joints 0.5 std
fig6, ax6 = plt.subplots()
j = np.array(joint_0_list_n)
j_std = np.array(0.5*joint_0_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_0_list_n)), joint_0_list_n, color='#FC800B', label="j_0")  # yellow
ax6.fill_between(range(len(joint_0_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax4.plot(range(len(joint_1_list_n)), joint_1_list_n, color='g', label="j_1")
j = np.array(joint_2_list_n)
j_std = np.array(0.5*joint_2_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_2_list_n)), joint_2_list_n, color='#D32826', label="j_2")  # red
ax6.fill_between(range(len(joint_2_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
# ax4.plot(range(len(joint_3_list_n)), joint_3_list_n, color='c', label="j_3")
j = np.array(joint_4_list_n)
j_std = np.array(0.5*joint_4_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_4_list_n)), joint_4_list_n, color="#187AB2", label="j_4")  # blue
ax6.fill_between(range(len(joint_4_list_n)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax4.plot(range(len(joint_5_list_n)), joint_5_list_n, color='y', label="j_5")
j = np.array(joint_6_list_n)
j_std = np.array(0.5*joint_6_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax6.plot(range(len(joint_6_list_n)), joint_6_list_n, color='#309F2D', label="j_6")  # green
ax6.fill_between(range(len(joint_6_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
# ax4.plot(range(len(joint_7_list_n)), joint_7_list_n, color='grey', label="j_7")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/b_joints_gerade_quater.pdf".format(walking_type, model_name), bbox_inches='tight')

fig7, ax7 = plt.subplots()
# ax5.plot(range(len(joint_0_list_n)), joint_0_list_n, color='b', label="j_0")
j = np.array(joint_1_list_n)
j_std = np.array(0.5*joint_1_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_1_list_n)), joint_1_list_n, color="#187AB2", label="j_1")  # blue
ax7.fill_between(range(len(joint_1_list_n)), j_plus, j_minus, alpha=0.5, facecolor="#187AB2")
# ax5.plot(range(len(joint_2_list_n)), joint_2_list_n, color='r', label="j_2")
j = np.array(joint_3_list_n)
j_std = np.array(0.5*joint_3_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_3_list_n)), joint_3_list_n, color='#309F2D', label="j_3")  # green
ax7.fill_between(range(len(joint_3_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#309F2D')
# ax5.plot(range(len(joint_4_list_n)), joint_4_list_n, color='m', label="j_4")
j = np.array(joint_5_list_n)
j_std = np.array(0.5*joint_5_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_5_list_n)), joint_5_list_n, color='#FC800B', label="j_5")  # yellow
ax7.fill_between(range(len(joint_5_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#FC800B')
# ax5.plot(range(len(joint_6_list_n)), joint_6_list_n, color='lime', label="j_6")
j = np.array(joint_7_list_n)
j_std = np.array(0.5*joint_7_std_list_n)
j_plus = j+j_std
j_minus = j-j_std
j_plus = list(j_plus)
j_minus = list(j_minus)
ax7.plot(range(len(joint_7_list_n)), joint_7_list_n, color='#D32826', label="j_7")  # red
ax7.fill_between(range(len(joint_7_list_n)), j_plus, j_minus, alpha=0.5, facecolor='#D32826')
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/b_joints_ungerade_quater.pdf".format(walking_type, model_name), bbox_inches='tight')

# c Walking Style Plots -- no std
fig8, ax8 = plt.subplots()
ax8.plot(range(len(forward_return_list1)), forward_return_list1, color="#187AB2", label="forward")
ax8.plot(range(len(electricity_list1)), electricity_list1, color='#D32826', label="electricity")
ax8.plot(range(len(joint_lim_list1)), joint_lim_list1, color='#FC800B', label="joint_limit")
ax8.plot(range(len(z_pos_list1)), z_pos_list1, color='#309F2D', label="z_pos")
plt.yticks(np.arange(-1, 2.5, step=0.5))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Performance")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/c_fejz_no_std.pdf".format(walking_type, model_name), bbox_inches='tight')

fig9, ax9 = plt.subplots()
ax9.plot(range(len(joint_0_list_n)), joint_0_list_n, color="#187AB2", label="j_0")
ax9.plot(range(len(joint_2_list_n)), joint_2_list_n, color='#D32826', label="j_2")
ax9.plot(range(len(joint_4_list_n)), joint_4_list_n, color='#FC800B', label="j_4")
ax9.plot(range(len(joint_6_list_n)), joint_6_list_n, color='#309F2D', label="j_6")
plt.yticks(np.arange(-45, 45, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/c_joints_gerade_no_std.pdf".format(walking_type, model_name), bbox_inches='tight')

fig10, ax10 = plt.subplots()
ax10.plot(range(len(joint_1_list_n)), joint_1_list_n, color="#187AB2", label="j_1")
ax10.plot(range(len(joint_3_list_n)), joint_3_list_n, color='#D32826', label="j_3")
ax10.plot(range(len(joint_5_list_n)), joint_5_list_n, color='#FC800B', label="j_5")
ax10.plot(range(len(joint_7_list_n)), joint_7_list_n, color='#309F2D', label="j_7")
plt.yticks(np.arange(0, 105, step=15))
plt.legend(loc="upper right")
plt.xlabel("Episodes")
plt.ylabel("Average Joint Angles")
plt.grid(True)
# plt.show()
plt.savefig("/Users/maxi/Desktop/Bachelor_Arbeit/BA_TUM/Models/proves"
            "/{}/{}/c_joints_ungerade_no_std.pdf".format(walking_type, model_name), bbox_inches='tight')


print("Mean joint 0: ", np.mean(joint_0_list_n))
print("Mean joint 1: ", np.mean(joint_1_list_n))
print("Mean joint 2: ", np.mean(joint_2_list_n))
print("Mean joint 3: ", np.mean(joint_3_list_n))
print("Mean joint 4: ", np.mean(joint_4_list_n))
print("Mean joint 5: ", np.mean(joint_5_list_n))
print("Mean joint 6: ", np.mean(joint_6_list_n))
print("Mean joint 7: ", np.mean(joint_7_list_n))

print("finished")
