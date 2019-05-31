import fetch_data_generation
import socket
import struct
import numpy as np
import gym

def collect_demos(collect_new=True):
  fileName = "data_fetch.npz"
  env = gym.make('FetchPickAndPlace-v1')
  if collect_new:
    actions, observations, info = fetch_data_generation.main()
    np.savez_compressed(fileName, acs=actions, obs=observations, info=info)
  else:
    demo_data = np.load(fileName, allow_pickle=True)
    actions, observations, info = demo_data['acs'], demo_data['obs'], demo_data['info']

  transitions = []
  # construct transitions
  for eps in zip(actions, observations, info):
    eps_acts = eps[0]
    eps_obs = eps[1]
    eps_infos = eps[2]
    for i in range(0, len(eps_acts)):
      nxt_transition = []
      nxt_transition += list(eps_obs[i]['observation'])
      nxt_transition += list(eps_obs[i]['desired_goal'])
      nxt_transition += list(eps_acts[i])
      nxt_transition += list(eps_obs[i+1]['observation'])
      nxt_transition += list(eps_obs[i+1]['desired_goal'])
      nxt_transition.append(0)
      nxt_transition.append(env.compute_reward(eps_obs[i+1]['achieved_goal'], eps_obs[i+1]['desired_goal'], eps_infos[i]))
      nxt_transition = [float(i) for i in nxt_transition]
      transitions.append(nxt_transition)
  
  return transitions

def simple_test():
  t1 = collect_demos()
  t2 = collect_demos(False)
  t1_len = len(t1)
  t2_len = len(t2)
  print("length {} {}".format(t1_len, t2_len))
  for i in range(t1_len):
    diff = [d1 for d1, d2 in zip(t1[i],t2[i]) if d1 != d2]
    if len(diff) > 0:
      print("i {}".format(diff))


def transfer_transitions():
  host = '127.0.0.1'
  port = 5555
  double_size = 8
  transtion_dim = 62
  count = 0
  info_dim = 1
  transitions = collect_demos(False)
  
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.bind((host, port))
  s.listen()
  print(len(transitions))
  print("Demo collector starts listening on {} ... ".format(port))
  conn, addr = s.accept()
  print("Demo collector established connection from {}".format(str(addr)))
  
  for t in transitions:
    data = conn.recv(info_dim*double_size)
    data = struct.unpack('d'*info_dim, data)
    if not data[0]:
      print("Receiver went wrong, aborting ...")
      s.close()
      exit(1)
    # print(len(t))
    message = struct.pack('d'*transtion_dim, *t)
    conn.sendall(message)
    count += 1
    print("sent {} transitions".format(count), end='\r')

  print("All data transferred, demo collector exiting ...")
  s.close()



transfer_transitions()
