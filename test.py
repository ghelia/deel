from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *
import time
deel = Deel()

CNN = GoogLeNet()
x=CNN.Input("deel.png")
start = time.clock()
CNN.classify(x)
ShowLabels()
