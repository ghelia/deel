from deel import *
from deel.network import *
from deel.commands import *
from deel.network.resnet152 import *
#from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel(gpu=-1)

CNN = ResNet152()

InputBatch(train="data/train.txt",
            val="data/test.txt")

def workout(x,t):
   CNN.batch_feature(x,t) 
   return CNN.backprop(t)

def checkout():
   CNN.save('model_google_cpu.hdf5')

BatchTrain(workout,checkout)
