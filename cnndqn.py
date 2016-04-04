from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *

deel = Deel()

CNN = AlexNet()

#CNN.Input("test.png")
#CNN.classify()
#ShowLabels()

def trainer(x,t):
	nin.classify(x)	
	return nin.backprop(t)

StartAgent(trainer)
