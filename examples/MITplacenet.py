from deel import *
from deel.network import *
from deel.network.caffenet import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()

net = CaffeNet(modelpath='googlelet_places205_train_iter_2400000.caffemodel',in_size=228,
			disableLayers=['loss1/ave_pool', 'loss2/ave_pool'],
			outputLayers=['loss3/classifier'],labels='misc/categoryIndex_places205.csv')

import sys
x=net.Input(sys.argv[1])
net.classify()
ShowLabels()
