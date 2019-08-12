import gym
import numpy as np
import scipy.misc
from skimage.transform import resize

np.set_printoptions(threshold=np.inf)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

class test_sim():
  def __init__(self, game='Pendulum-v0'):
    self.env = gym.make(game)
    
  
  def reset(self):
    observation = self.env.reset()
    # frame = self.env.render(mode='rgb_array')
    # frame_size = 512
    # start_p = int((frame.shape[0] - frame_size) / 2)
    # frame = frame[start_p:start_p+frame_size, start_p:start_p+frame_size, :].copy()
    # frame = resize(frame, (83, 83, 3))
    # frame = rgb2gray(frame)
    # frame = frame.flatten()
    # observation = np.concatenate((frame, observation))
    return observation

  def step(self, action, render=False):
    observation, reward, done, info = self.env.step(action)
    # frame = self.env.render(mode='rgb_array')
    # frame_size = 512
    # start_p = int((frame.shape[0] - frame_size) / 2)
    # frame = frame[start_p:start_p+frame_size, start_p:start_p+frame_size, :].copy()
    # frame = resize(frame, (83, 83, 3))
    # frame = rgb2gray(frame)
    # frame = frame.flatten()
    # observation = np.concatenate((frame, observation))
    # print(observation.shape)
    if render:
      self.env.render()
    return observation, reward, done, info

  def random_run(self):
    
    print(self.env.action_space.low)
    for _ in range(100000):
      observation = self.reset()
      counter = 0
      while 1:
        #print("observation {} {}".format(len(observation['observation']),observation))
        # for v in observation.values():
        #   a = [i for i in v if i >= 200 or i <= -200]
        #   print(len(a))
        #   if (len(a) > 0):
        #     exit(1)
        frame = self.env.render(mode='rgb_array')
        action = self.env.action_space.sample()
        # print("action {}".format(action))
        observation, reward, done, info = self.step(action)
        # print(observation)
        frame_size = 512
        start_p = int((frame.shape[0] - frame_size) / 2)
        frame = frame[start_p:start_p+frame_size, start_p:start_p+frame_size, :].copy()
        # print(frame[1:10,1:10,:])
        frame = resize(frame, (83, 83, 3))
        # print(frame[1:10,1:10,:])
        # print(frame)
        # for i in range(frame.shape[0]):
        #   for j in range(frame.shape[1]):
        #     if np.sum(frame[i][j]) < 200:
        #       for k in range(frame.shape[2]):
        #         frame[i][j][k] += 255

        # frame[:,:,0] = ((observation[2] + 8) / 16) * 255 + 0
        frame = rgb2gray(frame)
        print(frame[1:10, 1:10])
        scipy.misc.imsave('outfile'+str(counter)+'.jpg', frame)
        # counter += 1
        exit(1)

        # print(reward)
        # print("nxt observation {}".format(observation))
        # print("reward done info {} {} {}".format(reward, done, info))
        if done:
          break
  
  def close(self):
    self.env.close()

if __name__ == "__main__":
  # t = test_sim(game='FetchPickAndPlace-v1')
  t = test_sim(game='Pendulum-v0')

  t.random_run()
  t.close()