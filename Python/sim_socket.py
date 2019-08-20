import socket
import struct
from gym_sim import test_sim
import scipy
import numpy as np

class sim_socket:
  # 0 0 initialize
  # 0 1 step
  # 1 0 reset
  # 1 1 close
  def __init__(self, game='Pendulum-v0'):
    self.host = '127.0.0.1'
    self.port = 6666
    self.double_size = 4
    self.action_dim = 4 if game != 'Pendulum-v0' else 1
    self.state_dim = 31 if game != 'Pendulum-v0' else 3# + 83*83*1
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
      # print(len(data))
      data = struct.unpack('f'*(self.action_dim+self.flag_dim), data)
      #print("From C: {}".format(data))
      if not data[0] and not data[1]:
        #initialize or random action
        rnd_action = self.t.env.action_space.sample()
        reply = list(rnd_action) + [0] * (self.state_dim+self.info_dim-self.action_dim)
        reply = struct.pack('f'*(self.state_dim+self.info_dim), *reply)
        #print("Sent back random action: {}".format(rnd_action))
      if not data[0] and data[1]:
        #step
        self.count += 1
        action = data[self.flag_dim:]
        if (len(data)!=self.flag_dim+self.action_dim):
          print("Wrong data sent")
          break
        if self.count < 25000 or self.game != 'Pendulum-v0':
          observation, reward, done, _ = self.t.step(action)#, render=True)
        else:
          observation, reward, done, _ = self.t.step(action)#, render=True)
        terminate = 1.0 if done else 0.0
        #print("From C: a:{} r:{:.2f} d:{:.2f}".format(action,reward,terminate))
        if self.count % 1000 == 0:
          print("Stepped for {} times".format(self.count))
        if self.game == 'Pendulum-v0':
          flat_observation = list(observation)
        else:
          flat_observation = list(observation['observation']) + list(observation['desired_goal']) + list(observation['achieved_goal'])
        #observation = list(observation)
        # scipy.misc.imsave('step.jpg', np.array(flat_observation)[:-3].reshape(83,83))


        flat_observation.append(terminate)
        flat_observation.append(reward)
        # print("step sending {}".format(len(flat_observation)))

        #print(flat_observation)
        reply = struct.pack('f'*(self.state_dim+self.info_dim), *flat_observation)
      if data[0] and not data[1]:
        #reset
        observation = self.t.reset()
        #print(observation)
        if self.game == 'Pendulum-v0':
          flat_observation = list(observation)
        else:
          flat_observation = list(observation['observation']) + list(observation['desired_goal']) + list(observation['achieved_goal'])        #observation = list(observation)
        # scipy.misc.imsave('reset.jpg', np.array(flat_observation)[:-3].reshape(83,83))


        done, reward = 0, 0
        flat_observation.append(done)
        flat_observation.append(reward)
        # print("reset sending {}".format(len(flat_observation)))
        reply = struct.pack('f'*(self.state_dim+self.info_dim), *flat_observation)
      if data[0] and data[1]:
        #close
        still_open = False
        reply = struct.pack('f'*(self.state_dim+self.info_dim), *([0]*(self.state_dim+self.info_dim)))
      # sent = conn.send(reply)
      sent = self.sequential_send(conn, reply, self.state_dim+self.info_dim)
      # print("actually sent {}".format(sent))
    print("closing")

    #s.shutdown(socket.SHUT_RDWR)
    s.close()

  def sequential_send(self, conn, data, size):
    block_size = 512
    sent = 0
    
    data = struct.unpack('f'*(self.state_dim+self.info_dim), data)
    length = len(data)
    while sent < length:
      nxt_block = block_size if sent + block_size < length else length - sent
      # print("sending {} sent {} length {}".format(nxt_block, sent, length))
      reply = struct.pack('f'*nxt_block, *(data[sent:sent+nxt_block]))
      conn.sendall(reply)
      sent += nxt_block
    return sent


if __name__ == "__main__":
  new_socket = sim_socket()#game='FetchPickAndPlace-v1')
  new_socket.start_listen()
