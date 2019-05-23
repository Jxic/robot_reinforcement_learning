import gym

class test_sim():
  def __init__(self, game='Pendulum-v0'):
    self.env = gym.make(game)
    
  
  def reset(self):
    return self.env.reset()

  def step(self, action, render=False):
    n_state = self.env.step(action)
    if render:
      self.env.render()
    return n_state

  def random_run(self):
    
    print(self.env.action_space.low)
    for _ in range(100000):
      observation = self.reset()
      while 1:
        #print("observation {} {}".format(len(observation['observation']),observation))
        for v in observation.values():
          a = [i for i in v if i >= 200 or i <= -200]
          print(len(a))
          if (len(a) > 0):
            exit(1)
        self.env.render()
        action = self.env.action_space.sample()
        # print("action {}".format(action))
        observation, reward, done, info = self.step(action)
        # print(reward)
        # print("nxt observation {}".format(observation))
        # print("reward done info {} {} {}".format(reward, done, info))
        if done:
          break
  
  def close(self):
    self.env.close()

if __name__ == "__main__":
  t = test_sim(game='FetchPickAndPlace-v1')
  t.random_run()
  t.close()