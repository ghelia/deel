from deel import *
from deel.network import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()
BatchTrainer.batchsize=100

CNN = GoogLeNet()
dim = CNN.layerDim()


InputBatch(train="data/train.txt",
            val="data/test.txt")

def workout(x,t):
   CNN.batch_feature(x,t) 
   return CNN.backprop(t)

def checkout():
	CNN.save('modelcpu.hdf5')

BatchTrain(workout)