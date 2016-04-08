from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *
import deel.network.alexnet as alexnet

deel = Deel()

CNN = alexnet.AlexNet()
QNET = DQN(CNN.layerDim(u'pool5'),actions=[0,1,2,3])
QNET.min_eps = 0.2

def trainer(x):
	#CNN.classify(x)
	#ShowLabels()
	CNN.feature(x,layer='pool5')
	return QNET.actionAndLearn()

StartAgent(trainer)
