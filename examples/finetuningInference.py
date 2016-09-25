from deel import *
from deel.network import *
from deel.network.resnet152 import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()
BatchTrainer.batchsize=100

CNN = ResNet152(modelpath='misc/model_resnet_fc1000.hdf5',
			labels='data/labels.txt')

#CNN = GoogLeNet(modelpath='model_google_cpu2.hdf5',
#			labels='data/labels.txt')

import pickle

x=CNN.Input("test.jpg")
#CNN.Input("deel.png")
CNN.classify()
ShowLabels()
