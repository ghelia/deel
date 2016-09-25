from deel import *
from deel.network import *
from deel.commands import *
from deel.agentServer import *
from deel.network.googlenet import *
from deel.network.dqn import DQN

deel = Deel()

CNN = GoogLeNet()
CNN.ShowLayers()

QNET = DQN(CNN.layerDim(u'inception_5b/pool_proj'),actions=[0,1,2])

#Hyper Parameters
QNET.min_eps = 0.2
QNET.epsilonDelta = 1.0 / 10 ** 4
QNET.func.gamma = 0.99 # Discount factor
QNET.func.initial_exploration #10**4  # Initial exploratoin. original: 5x10^4
QNET.func.replay_size = 32 # Replay (batch) size
QNET.func.target_model_update_freq = 10**4  # Target update frequancy. original: 10^4
QNET.func.data_size = 10**5  # Data size of history. original: 10^6
QNET.func.hint_size = 1 #original: 4

def trainer(x):
	CNN.feature(x,layer='inception_5b/pool_proj')
	return QNET.actionAndLearn()

StartAgent(trainer)
