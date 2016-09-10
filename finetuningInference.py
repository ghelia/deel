from deel import *
from deel.network import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()
BatchTrainer.batchsize=100

CNN = GoogLeNet(modelpath='modelcpu.hdf5',labels='data/labels.txt')

import pickle

x=CNN.Input("test.jpg")
#CNN.Input("deel.png")
CNN.classify()
ShowLabels()
