
from deel import *

deel = Deel()

nin = NetworkInNetwork()


InputBatch(train="data/train.txt",
			val="data/test.txt")

def trainer(x,t):
	nin.classify(x)	
	return nin.backprop(t)

BatchTrain(trainer)

"""
CNN = GoogLeNet()

i = Input("deel.png")
CNN.classify()
ShowLabels()"""
