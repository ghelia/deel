
from deel import *
Deel.gpu=0

nin = NetworkInNetwork()

read_image("/home/deepstation/shi3z/deel/deel.png")

InputBatch(train="data/train.txt",
			val="data/test.txt")

def trainer(x,t):
	print "trainer"
	print x
	nin.classify(x)	
	nin.backprop(t)
	print "trainer_end"

BatchTrain(trainer)

"""
CNN = GoogLeNet()

i = Input("deel.png")
CNN.classify()
ShowLabels()"""
