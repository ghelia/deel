
from deel import *

nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
			val="data/test.txt")

def trainer(x,t):
	nin.classify(x)	
	return nin.loss(t)

BatchTrain(trainer)


CNN = GoogLeNet()
Input("deel.png")
CNN.classify()
ShowLabels()
