from deel import *
from deel.network import *
from deel.network.caffenet import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()

CNN = CaffeNet(modelpath='gender_net.caffemodel',in_size=228,
			outputLayers=['fc8'],labels=['male','female'])

import sys
x=CNN.Input(sys.argv[1])
CNN.classify()
ShowLabels()
