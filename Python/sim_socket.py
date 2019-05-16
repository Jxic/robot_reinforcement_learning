import socket
import struct
from gym_sim import test_sim

class sim_socket:
  # 0 0 initialize
  # 0 1 step
  # 1 0 reset
  # 1 1 close
  def __init__(self, game='pendulum'):
    self.host = '127.0.0.1'
    self.port = 6666
    self.double_size = 8
    self.action_dim = 4 if game != 'pendulum' else 1
    self.state_dim = 16 if game != 'pendulum' else 3
    self.flag_dim = 2
    self.info_dim = 2
    self.count =0
    self.t = test_sim(game=game)
    self.game = game

  def start_listen(self):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((self.host, self.port))
    s.listen()

    print("Starts listening ... ")
    conn, addr = s.accept()
    print("Established connection from {}".format(str(addr)))
    still_open = True
    while still_open:
      data = conn.recv((self.action_dim+self.flag_dim)*self.double_size)
      data = struct.unpack('d'*(self.action_dim+self.flag_dim), data)
      #print("From C: {}".format(data))
      if not data[0] and not data[1]:
        #initialize or random action
        rnd_action = self.t.env.action_space.sample()
        reply = list(rnd_action) + [0] * (self.state_dim+self.info_dim-self.action_dim)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *reply)
        #print("Sent back random action: {}".format(rnd_action))
      if not data[0] and data[1]:
        #step
        self.count += 1
        action = data[self.flag_dim:]
        if (len(data)!=self.flag_dim+self.action_dim):
          print("Wrong data sent")
          break
        if self.count < 25000 or self.game != 'pendulum':
          observation, reward, done, _ = self.t.step(action, render=True)
        else:
          observation, reward, done, _ = self.t.step(action)#, render=True)
        terminate = 1.0 if done else 0.0
        #print("From C: a:{} r:{:.2f} d:{:.2f}".format(action,reward,terminate))
        if self.count % 1000 == 0:
          print("Stepped for {} times".format(self.count))
        if self.game == 'pendulum':
          flat_observation = list(observation)
        else:
          flat_observation = list(observation['observation']) + list(observation['desired_goal']) + list(observation['achieved_goal'])
        #observation = list(observation)
        
        flat_observation.append(terminate)
        flat_observation.append(reward)
        #print(flat_observation)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *flat_observation)
      if data[0] and not data[1]:
        #reset
        observation = self.t.reset()
        #print(observation)
        if self.game == 'pendulum':
          flat_observation = list(observation)
        else:
          flat_observation = list(observation['observation']) + list(observation['desired_goal']) + list(observation['achieved_goal'])        #observation = list(observation)
        done, reward = 0, 0
        flat_observation.append(done)
        flat_observation.append(reward)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *flat_observation)
      if data[0] and data[1]:
        #close
        still_open = False
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *([0]*(self.state_dim+self.info_dim)))
      conn.sendall(reply)
    print("closing")

    #s.shutdown(socket.SHUT_RDWR)
    s.close()



if __name__ == "__main__":
  new_socket = sim_socket(game='FetchReach-v1')
  new_socket.start_listen()
