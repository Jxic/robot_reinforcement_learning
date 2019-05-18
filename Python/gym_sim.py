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
    observation = self.reset()
    print(self.env.action_space.low)
    for _ in range(100000):
      print("observation {} {}".format(len(observation['observation']),observation))
      self.env.render()
      action = self.env.action_space.sample()
      print("action {}".format(action))
      observation, reward, done, info = self.step(action)
      print("nxt observation {}".format(observation))
      print("reward done info {} {} {}".format(reward, done, info))
      # if done:
      #   break
  
  def close(self):
    self.env.close()

if __name__ == "__main__":
  t = test_sim(game='FetchReach-v1')
  t.random_run()
  t.close()