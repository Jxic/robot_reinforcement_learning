import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

'''
load in log file and parse into a dictionary
'''
def parse_log(path):
  data = {}
  data['rewards'] = []
  data['q'] = []
  reward_idx = 4
  q_idx = 15
  with open(path, 'r') as log:
    for line in log:
      strs = line.split()
      # print(len(strs))
      if (len(strs) != 21):
        continue
      data['q'].append(float(strs[q_idx]))
      data['rewards'].append(float(strs[reward_idx]))
  return data


def period_average(rewards):
  length = len(rewards)
  idx = 0
  batch = 10
  avg_rewards = []
  while idx + batch < length:
    nxt_batch = rewards[idx: idx+batch]
    mean = np.mean(np.array(nxt_batch))
    avg_rewards.append(mean)
    idx += batch
  last = rewards[idx: length]
  if len(last) > 0:
    mean = np.mean(np.array(last))
    avg_rewards.append(mean)
  return avg_rewards

def sample_half(rewards):
  ret = []
  for i in range(0, len(rewards), 2):
    ret.append(rewards[i])
  return ret
import random
'''
result is a dictionary with key being the type of data i.e. reward, Q, loss ...
'''
def plot_graph(results):

  # data_3 = parse_log("later_experiment1_3.0.log")
  # data_5 = parse_log("later_experiment1.log")

  data_3 = parse_log("pnp_run_without_mpi_success_rate.log")

  rewards_1_5 = results['q']
  rewards_3 = data_3['q']
  # rewards_5 = data_5['rewards']
  # rewards = data['q']
  # q = results['q']
  # rewards = rewards[:2527]
  # q = q[:2527]
  rewards_1_5 = rewards_1_5[:3000]
  rewards_3 = rewards_3[:3000]
  # rewards_5 = rewards_5[:30000]
  rewards_1_5 = period_average(rewards_1_5)
  rewards_3 = period_average(rewards_3)
  # rewards_5 = period_average(rewards_5)
  # q = period_average(q)
  # q = [0.9, 0.8, 0.4, 0.2,0.08,0.04, 0.06, 0, 0]
  # q = [0.9, 0.48,0.16, 0.16, 0.16 ,0.16, 0.16, 0,0]
  # rewards = sample_half(sample_half(rewards))
  # q = sample_half(sample_half(q))
  # for i in range(100, len(rewards)):
  #   if rewards[i] < 0.9:
  #     rewards[i] = random.randint(90, 100) / 100.0

  x_reward = np.linspace(0, len(rewards_3), len(rewards_3))
  x_reward_m = np.linspace(0, len(rewards_1_5), len(rewards_1_5))
  # x = np.linspace(0, len(q), len(q))
  
  # x = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]
  # x= [0, 0.005,0.007,0.05,0.1 ,0.5, 1, 1.5,2]
  plt.subplot(1, 1, 1)
  plt.plot(x_reward_m, rewards_1_5, label="distributed learning")
  plt.plot(x_reward, rewards_3, label='single-agent learning')
  # plt.plot(x_reward, rewards_5, label='5cm')

  plt.title("success rate comparison")
  plt.xlabel('episodes')
  plt.ylabel('average success rate')


  # plt.subplot(1, 2, 2)
  # plt.plot(x, q, color='orange')
  # plt.title("Q-value against episodes")
  # plt.xlabel('episodes')
  # plt.ylabel('mean Q-value')

  plt.legend()
  plt.show()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to the logfile", type=str, default="")
    return parser


if __name__ == '__main__':
  parser = build_argparser()
  args = parser.parse_args()

  print("Loading log file from " + args.path + " ... ")
  if args.path == "":
    print("Please provide a path for the log file")
    exit(1)
  
  plot_graph(parse_log(args.path))
  print("exiting ...")
