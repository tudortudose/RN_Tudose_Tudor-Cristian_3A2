import numpy


class Perceptron:
  
  def __init__(self, t):
    self.b = 0
    self.w = numpy.array([0 for x in range(784)])
    self.t = t
