import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import randint
import gym
from keras_networks import CriticNN, ActorNN
from tensorflow.keras.optimizers import Adam
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import pybulletgym

env = gym.make('Ant-v2')

