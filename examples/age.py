from deel import *
from deel.network import *
from deel.network.caffenet import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()

CNN = CaffeNet(modelpath='age_net.caffemodel',in_size=228,
			outputLayers=['fc8'],labels=['0-2','4-6','8-13','15-20','25-32','38-43','48-53','60-'])

import sys
x=CNN.Input(sys.argv[1])
CNN.classify()
ShowLabels()
