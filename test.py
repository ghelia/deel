
from deel import *
Deel.gpu=0

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
