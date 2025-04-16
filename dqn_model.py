import gymnasium as gym
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from threading import Thread
import time
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from IPython import display
from tensorflow.keras.utils import plot_model

env = gym.make("LunarLander-v3")

net_input = Input(shape=(8,))
x = Dense(64, activation="relu")(net_input)
x = Dense(64, activation="relu")(x)
x = Dense(64, activation="relu")(x)
output = Dense(4, activation="linear")(x)
q_net = Model(inputs=net_input, outputs=output)
q_net.compile(optimizer=Adam(learning_rate=0.001))
loss_fn = Huber()

target_net = clone_model(q_net)

# Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.005
GAMMA = 0.99
NUM_EPISODES = 600
MAX_TRANSITIONS = 1_00_000
TARGET_UPDATE_AFTER = 1000
LEARN_AFTER_STEPS = 4
BATCH_SIZE = 64

REPLAY_BUFFER = []


def insert_transition(transition):
    if len(REPLAY_BUFFER) >= MAX_TRANSITIONS:
        REPLAY_BUFFER.pop(0)
    REPLAY_BUFFER.append(transition)


def sample_transitions(batch_size=16):
    random_indices = tf.random.uniform(shape=(batch_size,), minval=0, maxval=len(REPLAY_BUFFER), dtype=tf.int32)
    sampled_current_states = []
    sampled_actions = []
    sampled_rewards = []
    sampled_next_states = []
    sampled_terminals = []

    for index in random_indices:
        sampled_current_states.append(REPLAY_BUFFER[index][0])
        sampled_actions.append(REPLAY_BUFFER[index][1])
        sampled_rewards.append(REPLAY_BUFFER[index][2])
        sampled_next_states.append(REPLAY_BUFFER[index][3])
        sampled_terminals.append(REPLAY_BUFFER[index][4])

    return tf.convert_to_tensor(sampled_current_states), tf.convert_to_tensor(sampled_actions), tf.convert_to_tensor(
        sampled_rewards, tf.float32), tf.convert_to_tensor(sampled_next_states), tf.convert_to_tensor(sampled_terminals)



def policy(state, explore=0.0):
    state = np.array(state, dtype=np.float32)              # ensure correct type/shape
    state_tensor = tf.convert_to_tensor(state)
    state_tensor = tf.expand_dims(state_tensor, axis=0)    # add batch dimension

    if tf.random.uniform(()) <= explore:
        action = tf.random.uniform((), maxval=4, dtype=tf.int32)
    else:
        action = tf.argmax(q_net(state_tensor)[0], output_type=tf.int32)

    return action


# Custom Reward Functions and Potential

def potential(state):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state

    # Distance from center
    dist = (x**2 + y**2) ** 0.5
    velocity = (vx**2 + vy**2) ** 0.5

    # Smaller values are better, so we subtract from max potential
    pos_score = 1.5 - dist       # closer to (0,0) is better
    vel_score = 1.5 - velocity   # slower is better
    angle_score = 1.0 - abs(angle)
    ang_vel_score = 1.0 - abs(ang_vel)

    # Clamp potential to be non-negative
    phi = max(0.0, pos_score + 0.5 * vel_score + 0.5 * angle_score + 0.5 * ang_vel_score)

    return phi

def reward_custom_1(state, action, done):

    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state

    reward = 0

    reward+=(1-abs(x))*5
    reward+=(1-y)*2
    if(y<0.5):
        reward+=-1.0*(abs(vx)+abs(vy))
        reward+=-1.0*abs(angle)
    
    if left_leg: reward+=5
    if right_leg: reward+=5
    if left_leg and right_leg: reward+=10

    if action==2: reward-=0.1
    elif action in [1,3]: reward-=0.02


    # Crash Penalty
    if done and (y<0.1 and abs(vy) >=0.1 and (not left_leg and not right_leg)):
        reward-= 100
    
    # Drift Away Penalty
    reward-=max(0, abs(x)-1.0) *50
    reward-=max(0, y-1.3) *20
    

    return reward


def reward_custom_pot(state, next_state, action, done, env_reward, gamma=0.99):
    

    # Compute potential values
    phi_s = potential(state)
    phi_s1 = potential(next_state)

    # Apply PBRS formula
    reward = env_reward + gamma * phi_s1 - phi_s

    # Optional: fuel efficiency shaping
    if action == 2:  # main engine
        reward -= 0.1
    elif action in [1, 3]:  # side engines
        reward -= 0.02

    return reward
    


def reward_custom_transit(state, next_state, action, done):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_leg, nright_leg = next_state

    # Compute relevant transition features
    d_t = (x ** 2 + y ** 2) ** 0.5
    d_t1 = (nx ** 2 + ny ** 2) ** 0.5

    v_t = (vx ** 2 + vy ** 2) ** 0.5
    v_t1 = (nvx ** 2 + nvy ** 2) ** 0.5

    omega_t = abs(ang_vel)
    omega_t1 = abs(nang_vel)

    reward = 0

    # Encourage approaching the landing pad
    reward -= 5 * (d_t1 - d_t)

    # Encourage reducing overall velocity
    reward -= 4 * (v_t1 - v_t)

    # Encourage reducing angular velocity
    reward -= 3 * (omega_t1 - omega_t)

    # Penalize falling too fast near ground
    if ny < 0.4:
        reward -= 2 * (0.4 - ny) * max(0, abs(nvy) - 0.3)

    # Encourage staying near center horizontally
    reward -= 2.5 * abs(nx)

    # Soft landing detection
    soft_landing = int(
        nleft_leg and nright_leg and
        abs(nvx) < 0.1 and abs(nvy) < 0.1 and
        abs(nangle) < 0.1 and abs(nang_vel) < 0.1
    )
    reward += 250 * soft_landing

    # Crash/hard landing penalties
    if done and ny < 0.1:
        if not soft_landing:
            reward -= 50

    # Alive bonus
    if not done:
        reward += 0.2

    return reward




# Training Step

random_states = []
done = False
i = 0
state, _ = env.reset()
state = np.array(state, dtype=np.float32)
while i < 20 and not done:
    random_states.append(state)
    state, _, te, tr, _ = env.step(policy(state).numpy())
    state = np.array(state, dtype=np.float32)
    done= te or tr
    i += 1

random_states = tf.convert_to_tensor(random_states)


def get_q_values(states):
    return tf.reduce_max(q_net(states), axis=1)

def check_soft_land(state):

    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
    return left_leg == 1 and right_leg == 1 and abs(vx) < 0.1 and abs(vy) < 0.1 and abs(angle)<0.1 and abs(ang_vel)<0.1

def check_landing(state):
    x, y, vx, vy, angle, ang_vel, left_leg, right_leg = state
    return left_leg == 1 and right_leg == 1 and  abs(vy) < 0.1 and abs(angle)<0.1

step_counter = 0
metric = {"episode": [], "length": [], "total_reward": [], "avg_q": [], "exploration": [], "land":[], "soft_land": []}
for episode in range(NUM_EPISODES):
    landing=False
    soft_landing=False
    done = False
    total_reward = 0
    episode_length = 0
    state, _ = env.reset()
    state = np.array(state, dtype=np.float32)
    while not done:
        action = policy(state, EPSILON)
        next_state, def_rew, te, tr, _ = env.step(action.numpy()) #use reward here for default reward
        next_state = np.array(next_state, dtype=np.float32)
        done=te or tr

        # Insert custom reward func below
        reward=reward_custom_pot(state, next_state, action, done, def_rew)
        soft_landing=check_soft_land(next_state)
        landing=check_landing(next_state)
        insert_transition([state, action, reward, next_state, done])
        state = next_state
        step_counter += 1

        if step_counter % LEARN_AFTER_STEPS == 0:
            current_states, actions, rewards, next_states, terminals = sample_transitions(BATCH_SIZE)

            next_action_values = tf.reduce_max(target_net(next_states), axis=1)

            # Bellman Eqn for target q values

            targets = tf.where(terminals, rewards, rewards + GAMMA * next_action_values)

            with tf.GradientTape() as tape:
                preds = q_net(current_states)
                batch_nums = tf.range(0, limit=BATCH_SIZE)
                indices = tf.stack((batch_nums, actions), axis=1)
                current_values = tf.gather_nd(preds, indices)
                loss = loss_fn(targets, current_values)

            grads = tape.gradient(loss, q_net.trainable_weights)
            q_net.optimizer.apply_gradients(zip(grads, q_net.trainable_weights))

        if step_counter % TARGET_UPDATE_AFTER == 0:
            target_net.set_weights(q_net.get_weights())

        total_reward += reward
        episode_length += 1

    EPSILON /= EPSILON_DECAY
    metric["episode"].append(episode)
    metric["length"].append(episode_length)
    metric["total_reward"].append(total_reward)
    metric["avg_q"].append(tf.reduce_mean(get_q_values(random_states)).numpy())
    metric["exploration"].append(EPSILON)
    metric["land"].append(landing)
    metric["soft_land"].append(soft_landing)

    pd.DataFrame(metric).to_csv("metric_pot.csv", index=False)
env.close()
q_net.save("dqn_q_net_pot.keras")