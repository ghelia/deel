from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *
import time
deel = Deel(gpu=0)

CNN = GoogLeNet()
x=CNN.Input("deel.png")
start = time.clock()
print "start"
for i in range(100):
	CNN.classify(x)
end = time.clock()
print start-end
ShowLabels()
