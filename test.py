
from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

InputBatch(train="data/train_lstm.tsv")

CNN = GoogLeNet()
RNN = LSTM()

def trainer(x,t):
	CNN.classify() 
	RNN.learn(t)
	return RNN.backprop()

BatchTrain(trainer)

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
CNN = GoogLeNet()

i = Input("deel.png")
CNN.classify()
ShowLabels()
"""
