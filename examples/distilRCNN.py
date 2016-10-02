from deel import *
from deel.network.rnin import *
from deel.tensor import *
from deel.commands import *
from deel.network.googlenet import *
from deel.network.fasterRCNN import FasterRCNN

deel = Deel(gpu=0)
Deel.epoch=10000
student = RegionalNetworkInNetwork(labels="data/labels.txt")
teacher = FasterRCNN()

InputBatch(train="../deel/data/train.txt",
			val="../deel/data/test.txt",
			minibatch=False)
BatchTrainer.batchsize=1
def workout(x,t):
	scores,box_deltas = teacher.feature(x)
	num_of_dummys = 300 - scores.data.shape[0]
	t = concat(scores.data,np.zeros(21*num_of_dummys))
	t = concat(t,box_deltas)
	t = concat(t,np.zeros(21*4*num_of_dummys))
	t = ChainerTensor(chainer.Variable(Deel.xp.asarray(t,dtype=np.float32), volatile='off'))
	x = ChainerTensor(chainer.Variable(Deel.xp.asarray([x.value],dtype=np.float32), volatile='off'))
	student.classify(x,train=True)	
	return student.backprop(t,distill=True)

def checkout():
   student.save('rcnn2rnin_wisky.hdf5')

BatchTrain(workout,checkout)
