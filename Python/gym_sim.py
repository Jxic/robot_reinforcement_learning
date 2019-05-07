import gym

class test_sim():
  def __init__(self):
    self.env = gym.make('Pendulum-v0')
    
  
  def reset(self):
    return self.env.reset()

  def step(self, action, render=False):
    n_state = self.env.step(action)
    if render:
      self.env.render()
    return n_state

  def random_run(self):
    observation = self.reset()
    for _ in range(1000):
      print(observation)
      self.env.render()
      action = self.env.action_space.sample()
      observation, reward, done, info = self.step(action)
      print(action)
      print("reward done info {} {} {}".format(reward, done, info))
      if done:
        break
  
  def close(self):
    self.env.close()

if __name__ == "__main__":
  t = test_sim()
  t.random_run()
  t.close()