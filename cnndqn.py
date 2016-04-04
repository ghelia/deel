from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *

deel = Deel()

CNN = AlexNet()
#CNN.Input("test.png")
#CNN.classify()
#ShowLabels()
QNET = DQN()

def trainer(x):
	CNN.classify(x)
	ShowLabels()
	CNN.feature(x)
	return QNET.actionAndLearn()

StartAgent(trainer)
