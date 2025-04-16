import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd
import os
from threading import Thread
import time

SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)  # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
plt.rc('figure', titlesize=SMALL_SIZE)  # fontsize of the figure title

fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
fig4 = plt.figure(4)
fig5 = plt.figure(5)
fig6 = plt.figure(6)

episodic_data = None

AGENT_TRAIN_METRIC_PATH = "metric_pot.csv"  # Metric file path


# Reading our metric on a different thread
def read_data():
    global episodic_data
    while True:
        try:
            episodic_data = pd.read_csv(AGENT_TRAIN_METRIC_PATH)
        except Exception as e:
            print(e)
        time.sleep(10)


t = Thread(target=read_data)
t.start()


# Functions to plot graphs

def plot(i):
    try:
        plt.figure(1)
        plt.cla()
        plt.plot(episodic_data["episode"], episodic_data['total_reward'], linewidth=1, color="#2AD4FF")
        plt.title("Total Reward per episode")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.grid(True)
    except Exception as e:
        print(e)


def plot2(i):
    try:
        plt.figure(2)
        plt.cla()
        plt.plot(episodic_data["episode"], episodic_data['avg_q'], linewidth=1, color="#FF00FF")
        plt.title("Average Q")
        plt.xlabel("Episode")
        plt.ylabel("Q Value")
        plt.grid(True)
    except Exception as e:
        print(e)


def plot3(i):
    try:
        plt.figure(3)
        plt.cla()
        plt.plot(episodic_data["episode"], episodic_data['length'], linewidth=1, color="#00FF00")
        plt.title("Length of episode")
        plt.xlabel("Episode")
        plt.ylabel("Length")
        plt.grid(True)
    except Exception as e:
        print(e)


def plot4(i):
    try:
        plt.figure(4)
        plt.cla()
        plt.plot(episodic_data["episode"], episodic_data['exploration'], color="#FF6600")
        plt.title("Exploration")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
    except Exception as e:
        print(e)

def plot5(i):
    try:
        plt.figure(5)
        plt.cla()
        plt.scatter(episodic_data["episode"], episodic_data['land'], color="blue")
        plt.title("Successful Landing")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
    except Exception as e:
        print(e)

def plot6(i):
    try:
        plt.figure(6)
        plt.cla()
        plt.scatter(episodic_data["episode"], episodic_data['soft_land'], color="red")
        plt.title("Successful Soft Landing")
        plt.xlabel("Episode")
        plt.ylabel("Epsilon")
        plt.grid(True)
    except Exception as e:
        print(e)


# Graphs are updated every 5 seconds
anim = FuncAnimation(fig1, plot, interval=5000)
anim2 = FuncAnimation(fig2, plot2, interval=5000)
anim3 = FuncAnimation(fig3, plot3, interval=5000)
anim4 = FuncAnimation(fig4, plot4, interval=5000)
anim5 = FuncAnimation(fig5, plot5, interval=5000)
anim6 = FuncAnimation(fig6, plot6, interval=5000)

plt.show()