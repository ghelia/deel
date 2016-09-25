from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *
from deel.network.alexnet import AlexNet
from deel.network.dqn import DQN

deel = Deel()

CNN = AlexNet()
QNET = DQN()

def trainer(x):
	CNN.feature(x)
	return QNET.actionAndLearn()

StartAgent(trainer)
