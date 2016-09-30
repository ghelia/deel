from deel import *
from deel.network import *
from deel.network.resnet152 import *
from deel.commands import *
from deel.network.googlenet import *
from deel.network.nin import *
import chainer.functions as F
import time

deel = Deel()

#CNN = ResNet152(modelpath='misc/model_resnet_fc1000.hdf5',
#			labels='data/labels.txt')

CNN = NetworkInNetwork(modelpath='google_nin_wisky.hdf5',
			labels='misc/labels.txt')

import pickle
import sys
x=CNN.Input(sys.argv[1])
CNN.classify()
ShowLabels()
