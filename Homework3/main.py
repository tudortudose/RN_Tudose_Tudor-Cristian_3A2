from copyreg import pickle
import pickle, gzip
from network import Network

def read_data_sets(file_name):
  with gzip.open(file_name, 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')
  return train_set, valid_set, test_set


def run():
  train_set, valid_set, test_set = read_data_sets('mnist.pkl.gz')
  network = Network(784, 100, 10)
  network.train(train_set, 0.01, 10)
  network.test(test_set)

run()