import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np



'''
load in log file and parse into a dictionary
'''
def parse_log(path):
  data = {}
  data['total'] = []
  data['prep'] = []
  data['forward'] = []
  data['backward'] = []
  data['update'] = []
  data['loss'] = []
  total_idx = 4
  loss_idx = 2
  prep_idx = 8
  forward_idx = 10
  backward_idx = 12
  update_idx = 14
  with open(path, 'r') as log:
    for line in log:
      strs = line.split()
      print(strs)
      if (len(strs) != 15):
        continue
      data['prep'].append(float(strs[prep_idx]))
      data['forward'].append(float(strs[forward_idx]))
      data['backward'].append(float(strs[backward_idx]))
      data['update'].append(float(strs[update_idx]))
      data['loss'].append(float(strs[loss_idx]))
      data['total'].append(float(strs[total_idx]))
    
  return data


def plot_graph(results):
  prep = results['prep']
  forward = results['forward']
  backward = results['backward']
  update = results['update']
  loss = results['loss']
  total = results['total']

  x = np.linspace(0, 99, 100)

  plt.subplot(1, 2, 1)
  plt.plot(x, loss)
  plt.title('loss against epoch')
  plt.xlabel('epoch')
  plt.ylabel('loss')

  plt.subplot(1,2,2)
  lines = plt.plot(x, total, 'r', x, prep, 'b--', x, forward, 'r--', x , backward, 'g--', x, update, 'm--')
  plt.legend(lines, ['Total Time', 'Preparation time', 'Forward prop. time', 'backward prop. time', 'update time'])
  plt.title('time taken against epoch')
  plt.xlabel('epoch')
  plt.ylabel('time / ms')

  plt.show()

def comparision_graph(results):
  prep = results['prep']
  forward = results['forward']
  backward = results['backward']
  update = results['update']
  loss = results['loss']
  total = results['total']

  x = np.linspace(100, 900, 9)

  keras_res = []
  with open("keras_log", 'r') as log:
    for line in log:
      keras_res = line.split()

  keras_res = np.array(keras_res).astype(float)

  print(keras_res)
  keras_res *= 1000.0
  total_ = []
  for i in range(9):
    total_.append(np.sum(total[i*100: i*100+100]))

  plt.subplot(1, 1, 1)
  print(len(total))
  plt.plot(x, keras_res, label='keras')
  plt.plot(x, total_, label='ours')
  plt.legend()
  plt.title('time against number of neurons in layers')
  plt.xlabel('number of neurons')
  plt.ylabel('time / ms')

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
  
  # plot_graph(parse_log(args.path))
  comparision_graph(parse_log(args.path))
  # parse_log(args.path)
  print("exiting ...")
