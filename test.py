
from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN = GoogLeNet()

CNN.Input("deel.png")
CNN.classify()
Show()
ShowLabels()


"""
InputBatch(train="data/train_lstm.tsv")

CNN = GoogLeNet()
RNN = LSTM()

def trainer(x,t):
	CNN.classify(x) 
	RNN.learn(t)
	return RNN.backprop()

BatchTrain(trainer)
"""
"""
nin = NetworkInNetwork()

InputBatch(train="data/train.txt",
			val="data/test.txt")
def trainer(x,t):
	nin.classify(x)	
	return nin.backprop(t)

BatchTrain(trainer)
"""
"""
"""
