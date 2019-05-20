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
  reward_idx = 4
  with open(path, 'r') as log:
    for line in log:
      strs = line.split()
      if (len(strs) != 16):
        continue
      data['rewards'].append(float(strs[reward_idx]))
  return data


def period_average(rewards):
  length = len(rewards)
  idx = 0
  batch = 100
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

'''
result is a dictionary with key being the type of data i.e. reward, Q, loss ...
'''
def plot_graph(results):
  fig,ax = plt.subplots()
  rewards = results['rewards']
  rewards = period_average(rewards)
  x = range(1, len(rewards)+1)
  ax.plot(x, rewards)
  plt.xticks(np.arange(min(x)-1, max(x), 10000))
  ax.set(xlabel='episodes', ylabel='rewards')
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
