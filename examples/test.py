from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.network.resnet152 import *
from deel.commands import *
import time
deel = Deel()

#CNN = ResNet152()
CNN = GoogLeNet()
CNN.Input("test.jpg")
CNN.classify()
ShowLabels()
