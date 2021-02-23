import numpy as np
import gym
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_networks import ActorNN, CriticNN
import tensorflow as tf
from replay_buffer import ReplayBuffer
from noise import NormalActionNoise
import matplotlib.pyplot as plt
import sys
import os
import pybulletgym

#reward_fcn_name = sys.argv[1]
reward_fcn_name = "test"


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
    for e in range(episode):

        # receive initial observation state s1 (observation = s1)
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
            reward_things = env.env.rewards
            reward = reward_things[1]   # progress

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

            if done:
                performance.append(score)
                avg_reward = np.mean(performance[-10:])
                avg_return.append(avg_reward)
                cumulus_steps += i
                print("episode: {}/{}, score: {}, avg_score: {}, ep_steps: {}, cumulus_steps: {}"
                      .format(e, episode, score, avg_reward, i, cumulus_steps))

                if 500000 < cumulus_steps < 501000:
                    if not os.path.exists("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name)):
                        os.mkdir("/home/ga53cov/Bachelor_Arbeit/BA/Models/Ant_v2/{}".format(reward_name))
                    q1.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                    "Models/Ant_v2/{}/q1{}.h5".format(reward_name, cumulus_steps))
                    q2.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                    "Models/Ant_v2/{}/q2{}.h5".format(reward_name, cumulus_steps))
                    q1_target.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                           "Models/Ant_v2/{}/q1t{}.h5".format(reward_name, cumulus_steps))
                    q2_target.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                           "Models/Ant_v2/{}/q2t{}.h5".format(reward_name, cumulus_steps))
                    mu.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                    "Models/Ant_v2/{}/mu{}.h5".format(reward_name, cumulus_steps))
                    mu_target.save_weights("/home/ga53cov/Bachelor_Arbeit/BA/"
                                           "Models/Ant_v2/{}/mut{}.h5".format(reward_name, cumulus_steps))
                    np.save("/home/ga53cov/Bachelor_Arbeit/BA/"
                            "Models/Ant_v2/{}/array{}".format(reward_name, cumulus_steps), avg_return)
                break

            score += reward
            state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        # stop learning after certain time steps
        if cumulus_steps > breaking_step:
            break

    return avg_return, mu


def test(mu_render, e, train_bool, weight_string):
    env = gym.make("Ant-v2")
    if not train_bool:
        mu_render.load_weights(weight_string)

    for i in range(e):
        observation = env.reset()
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        done = 0
        ep_reward = 0
        step = 0
        while not done:
            env.render()
            action = mu_render(state)
            next_state, reward, done, _ = env.step(action)
            state = tf.convert_to_tensor([next_state], dtype=tf.float32)
            ep_reward += reward
            step += 1
        print(ep_reward)
        print(step)


# main starts
train = True
break_step = 502000
agent_weights = "none"

if not train:
    break_step = 2000
    agent_weights = "/Users/maxi/Desktop/Bachelor_Arbeit/BA_Luca_Rep/BA/Models/Ant_v2/default_r/mu_5497.h5"

episodes = 500000
overall_performance, mu = ddpg(episodes, break_step, reward_fcn_name)

# plot performance
if train:
    plt.plot(range(len(overall_performance)), overall_performance, 'b')
    plt.title("Avg Test Reward Vs Test Episodes")
    plt.xlabel("Test Episodes")
    plt.ylabel("Average Test Reward")
    plt.grid(True)
    # plt.show()
    plt.savefig("/home/ga53cov/Bachelor_Arbeit/BA/"
                "Models/Ant_v2/{}/figure.pdf".format(reward_fcn_name), bbox_inches='tight')

# test and render
eps = 20
if not train:
    test(mu, eps, train, agent_weights)
