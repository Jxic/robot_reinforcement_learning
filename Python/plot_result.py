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
  q_idx = 11
  with open(path, 'r') as log:
    for line in log:
      strs = line.split()
      # print(len(strs))
      if (len(strs) != 17):
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
    mean = np.median(np.array(nxt_batch))
    avg_rewards.append(mean)
    idx += batch
  last = rewards[idx: length]
  if len(last) > 0:
    mean = np.median(np.array(last))
    avg_rewards.append(mean)
  return avg_rewards

def sample_half(rewards):
  ret = []
  for i in range(0, len(rewards), 2):
    ret.append(rewards[i])
  return ret

'''
result is a dictionary with key being the type of data i.e. reward, Q, loss ...
'''
def plot_graph(results):
  
  rewards = results['rewards']
  q = results['q']
  rewards = period_average(rewards)
  q = period_average(q)
  # rewards = sample_half(sample_half(rewards))
  # q = sample_half(sample_half(q))
  x_reward = np.linspace(0, len(rewards), len(rewards))
  x = np.linspace(0, len(q), len(q))
  plt.subplot(1, 2, 1)
  plt.plot(x_reward, rewards)
  plt.title("rewards against episodes")
  plt.xlabel('episodes')
  plt.ylabel('rewards')
  plt.subplot(1, 2, 2)
  plt.plot(x, q, color='orange')
  plt.title("Q-value against episodes")
  plt.xlabel('episodes')
  plt.ylabel('mean Q-values')
  # plt.xticks(np.arange(min(x)-1, max(x), 10000))
  # ax.set(xlabel='episodes', ylabel='rewards')
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
