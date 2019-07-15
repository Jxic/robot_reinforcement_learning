import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np

'''
load in log file and parse into a dictionary
'''
def parse_log(path):
  data = []

  with open(path, 'r') as log:
    nxt_tuple = []
    curr_point = 1
    for line in log:
      strs = line.split()
      # print(len(strs))
      if (len(strs) != 4):
        continue
      if curr_point % 3 == 0:
        nxt_tuple.append(convert_to_float(strs))
        # print("finished on tuple")
        # print(nxt_tuple)
        data.append(nxt_tuple)
        curr_point += 1
        nxt_tuple = []
        continue
      nxt_tuple.append(convert_to_float(strs))
      # print(nxt_tuple)
      curr_point += 1
  return data


def convert_to_float(strs):
  res = []
  for i in range(1, len(strs)):
    res.append(float(strs[i]))

  # print(res)
  return res

def organize_errs_into_grid(results):
  rows = 3
  cols = 4

  results = np.array(results)

  sim_errs = np.array([[0.0]*cols for _ in range(rows)])
  act_errs = np.array([[0.0]*cols for _ in range(rows)])
  counts = np.array([[0.0]*cols for _ in range(rows)])
  
  # for i in range(rows):
  #   for j in range(cols):

  for r in results:
    sim = r[0]
    act = r[1]
    tar = r[2]
    sim_err = np.sqrt(np.sum((sim-tar)**2)) #np.sum(np.absolute(sim-tar))
    act_err = np.sqrt(np.sum((act-tar)**2))#np.sum(np.absolute(act-tar))
    x = tar[0]
    z = tar[2]
    w_r = 4 - int(z/5)
    w_c = 3 - int((x+10)/5)
    print("writing to {}".format((w_r, w_c)))
    sim_errs[w_r][w_c] += sim_err
    act_errs[w_r][w_c] += act_err
    counts[w_r][w_c] += 1
  avg_sim_errs = sim_errs / counts
  avg_act_errs = act_errs / counts
  print(sim_errs, act_errs, counts)
  print(avg_sim_errs, avg_act_errs)
  return avg_sim_errs, avg_act_errs


'''
result is a dictionary with key being the type of data i.e. reward, Q, loss ...
'''
def plot_graph(results):
  print(results)
  print("start")
  for r in results:
    sim = r[0]
    act = r[1]
    tar = r[2]
  
    print("tar: ({:0.2f}, {:0.2f}, {:0.2f})".format(tar[0], tar[1], tar[2]))
    print("sim: ({:0.2f}, {:0.2f}, {:0.2f})".format(sim[0], sim[1], sim[2]))
    print("act: ({:0.2f}, {:0.2f}, {:0.2f})".format(act[0], act[1], act[2]))
  print("end")
  avg_sim_errs, avg_act_errs = organize_errs_into_grid(results)
  rows = 3
  cols = 4

  

  x_min, x_max, y_min, y_max = -10, 10, 10, 20

  # plt.subplot(1,1,1)
  # plt.imshow(avg_sim_errs, extent=(x_min, x_max, y_min, y_max) ,interpolation='gaussian')
  # plt.colorbar(orientation="horizontal")
  # plt.title("Average errors in simulation")
  # plt.xlabel("width / cm")
  # plt.ylabel("depth / cm")

  # plt.subplot(1,1,1)
  # plt.imshow(avg_act_errs, extent=(x_min, x_max, y_min, y_max), interpolation='gaussian')
  # plt.colorbar(orientation="horizontal")
  # plt.title("Average errors in real world")
  # plt.xlabel("width / cm")
  # plt.ylabel("depth / cm")
  
  # plt.show()



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
