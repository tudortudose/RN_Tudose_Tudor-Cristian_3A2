from copyreg import pickle
import pickle, gzip, numpy
from perceptron import Perceptron


def activation(input):
  return input > 0


def train_one(data_set, perceptron, rate, nrOperations):
  print('Train perceptron ', perceptron.t)
  all_classified = False
  size = len(data_set[0])
  totalOperations = nrOperations

  while(not(all_classified) and nrOperations > 0):
    print("round: ", totalOperations - nrOperations)
    all_classified = True
    for i in range(size):
      x = data_set[0][i]
      t = data_set[1][i]
      z = numpy.dot(perceptron.w, x) + perceptron.b
      output = activation(z)
      result = 0
      
      if perceptron.t == t:
        result = 1

      perceptron.w = perceptron.w + (result - output) * x * rate
      perceptron.b = perceptron.b + (result - output) * rate
      if output!=result:
        all_classified = False
    nrOperations -= 1
   

def train_all(data_set, perceptrons):  
  rate = 0.01
  nrOperations = 10
  for perceptron in perceptrons:
    train_one(data_set, perceptron, rate, nrOperations)
  return perceptrons


def test_all(data_set, perceptrons):
  size = len(data_set[0])
  wrong = 0
  for i in range(size):
    x = data_set[0][i]
    t = data_set[1][i]
    z = {}
    output = {}
    for perceptron in perceptrons:
      z[perceptron] = numpy.dot(perceptron.w, x) + perceptron.b
      output[perceptron] = activation(z[perceptron])
    p_max = max(z, key=z.get)
    if p_max.t != t:
      wrong += 1
  return (size-wrong)*100/size
 

def run():
  with gzip.open('mnist.pkl.gz', 'rb') as fd:
    train_set, valid_set, test_set = pickle.load(fd, encoding='latin')

  trained_perceptrons = train_all(train_set, [Perceptron(i) for i in range(10)])
  accuracy = test_all(test_set, trained_perceptrons)

  print("Accuracy: ", accuracy)


run()