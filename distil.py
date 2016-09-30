from deel import *
from deel.network.nin import *
from deel.commands import *
from deel.network.googlenet import *

deel = Deel(gpu=1)
Deel.epoch=10000
student = NetworkInNetwork(labels="data/labels.txt")
teacher = GoogLeNet(modelpath="bvlc_googlenet.caffemodel",
					labels="data/labels.txt")

InputBatch(train="../deel/data/train.txt",
			val="../deel/data/test.txt")
def workout(x,t):
	t = teacher.batch_feature(x)
	student.classify(x)	
	return student.backprop(t,distill=True)

def checkout():
   student.save('google_nin_wisky.hdf5')

BatchTrain(workout,checkout)
