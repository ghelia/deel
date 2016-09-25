from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *
from deel.network.alexnet import AlexNet
from deel.network.dqn import DQN

deel = Deel()

CNN = AlexNet()
QNET = DQN(CNN.layerDim(u'pool5'),depth_image_dim=Depth_dim)

def trainer(x):
	CNN.feature(x,layer=u'pool5')
	Concat(DepthImage())
	return QNET.actionAndLearn()

StartAgent(trainer)
