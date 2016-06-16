from deel import *
from deel.network import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel(gpu=0)


CNN = GoogLeNet()
dim = CNN.layerDim()
print dim
d = 1
for a in dim:
	d *= a

dim = d
print dim

p = Perceptron(layers=(dim,101),activation=F.sigmoid)

InputBatch(train="data/train.txt",
            val="data/test.txt")

def workout(x,t):
    f = CNN.batch_feature(x) 
    f.content.data = Deel.xp.asarray(f.content.data.reshape((32,dim)),dtype=Deel.xp.float32)
    p.forward()
    return p.backprop(t)

BatchTrain(workout)

