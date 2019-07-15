import gym_sim
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten

from keras import Model
from keras import backend as K
from keras import 

height = 83
width = 83
channel = 1
full_state_dim = 3
action_dim = 1
gamma = 0.99
c_lr = 0.001
a_lr = 0.0001
epoch = 10000
polyak = 0.999
batch_size = 128
pre_train_steps 256
memory_size = 1000000
noise_scale = 0.1
action_bound = 2


class Actor():

  def __init__(self, state_dim, action_dim):
    self.create_model((height, width, channel), action_dim)
    self.build_train_fn()

  def create_model(self, state_dim, action_dim):
    self.X = Input(shape=state_dim)
    conv1 = 
    return

  def build_train_fn(self):
    return

  def fit(self, batch):
    return

  def get_action(self, model, state, noise_scale):
    return

class Critic():
  
  def __init__(self, state_dim, action_dim):
    sefl