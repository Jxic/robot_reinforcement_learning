import socket
import struct
from gym_sim import test_sim

class sim_socket:
  # 0 0 initialize
  # 0 1 step
  # 1 0 reset
  # 1 1 close
  def __init__(self):
    self.host = '127.0.0.1'
    self.port = 6666
    self.double_size = 8
    self.action_dim = 1
    self.state_dim = 3
    self.flag_dim = 2
    self.info_dim = 2
    self.count =0
    self.t = test_sim()

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
        rnd_action = self.t.env.action_space.sample()
        reply = [rnd_action[0]] * (self.state_dim+self.info_dim)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *reply)
        #print("Sent back random action: {}".format(rnd_action[0]))
      if not data[0] and data[1]:
        self.count += 1
        action = data[self.flag_dim:]
        if (len(data)!=self.flag_dim+self.action_dim):
          print("Wrong data sent")
          break
        if self.count < 25000:
          observation, reward, done, _ = self.t.step(action)
        else:
          observation, reward, done, _ = self.t.step(action, render=True)
        terminate = 1.0 if done else 0.0
        if reward > -3:
          print("From C: a:{:.4f} r:{:.2f} d:{:.2f}".format(action[0],reward,terminate))
        if self.count % 10000 == 0:
          print("Stepped for {} times".format(self.count))
        observation = list(observation)
        
        observation.append(terminate)
        observation.append(reward)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *observation)
      if data[0] and not data[1]:
        observation = self.t.reset()
        observation = list(observation)
        done, reward = 0, 0
        observation.append(done)
        observation.append(reward)
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *observation)
      if data[0] and data[1]:
        still_open = False
        reply = struct.pack('d'*(self.state_dim+self.info_dim), *([0]*(self.state_dim+self.info_dim)))
      conn.sendall(reply)
    print("closing")

    #s.shutdown(socket.SHUT_RDWR)
    s.close()



if __name__ == "__main__":
  new_socket = sim_socket()
  new_socket.start_listen()
