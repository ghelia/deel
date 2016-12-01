from deel import *
from deel.network import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()

CNN = GoogLeNet()

InputBatch(train="data/train.txt",
            val="data/test.txt")

def workout(x,t):
   CNN.batch_feature(x) 
   return CNN.backprop(t)

def checkout():
   CNN.save('model_google_cpu.hdf5')

BatchTrain(workout,checkout)
