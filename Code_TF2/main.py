import numpy as np
import gym
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from keras_networks import ActorNN, CriticNN
import tensorflow as tf
from replay_buffer import ReplayBuffer
from noise import NormalActionNoise


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


def ddpg(episode):
    env = gym.make('Walker2d-v3')

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

            # execute action a_t and observe reward, and next state
            next_state, reward, done, _ = env.step(action)

            # store transition in replay buffer
            replay_buffer.store_transition(state, action, reward, next_state, done)

            # if there are enough transitions in the replay buffer
            batch_size = 64
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
                print("episode: {}/{}, score: {}".format(e, episode, score))
                break

            score += reward
            state = tf.convert_to_tensor([next_state], dtype=tf.float32)

        print("episode: {}/{}, score: {}".format(e, episode, score))
        performance.append(score)
    return performance


# main starts
episodes = 400
overall_performance = ddpg(episodes)

# plot performance
