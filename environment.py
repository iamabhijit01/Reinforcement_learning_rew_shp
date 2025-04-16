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

env = gym.make("LunarLander-v3", render_mode="rgb_array")
q_net = load_model("dqn_q_net.keras")
os.makedirs("videos", exist_ok=True)

def policy(state, explore=0.0):
    action = tf.argmax(q_net(state)[0], output_type=tf.int32)
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        action = tf.random.uniform(shape=(), minval=0, maxval=4, dtype=tf.int32)
    return action


fps = 60
video_filename = "videos/lunar_lander_default.mp4"
video_writer = None
env.reset()
first_frame = env.render()
height, width, _ = first_frame.shape
video_writer = cv2.VideoWriter(
        video_filename,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
video_writer.write(cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))


for episode in range(5):
    done = False
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)
    
    while not done:
        frame = env.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.imshow("Lunar Lander", frame)
        cv2.waitKey(10)

        action = policy(state)
        #action = env.action_space.sample() ## random agent
        next_state, reward, te, tr, _ = env.step(action.numpy()) #action.numpy() for trained model
        done= te or tr
        state = tf.convert_to_tensor([next_state])
env.close()
video_writer.release()
cv2.destroyAllWindows()
print(f"Video saved as {video_filename}")