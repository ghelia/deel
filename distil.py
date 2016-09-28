from deel import *
from deel.network.nin import *
from deel.commands import *
from deel.network.googlenet import *

deel = Deel(gpu=-1)

student = NetworkInNetwork(labels="../deel/data/labels.txt")
teacher = GoogLeNet(modelpath="../deel/misc/bvlc_googlenet.caffemodel",
					labels="../deel/data/labels.txt")

InputBatch(train="../deel/data/train.txt",
			val="../deel/data/test.txt")
def workout(x,t):
	print x.value.shape
	t = teacher.batch_feature(x)
	student.classify(x)	
	return student.backprop(t,distill=True)

def checkout():
   CNN.save('google_nin_wisky.hdf5')

BatchTrain(workout,checkout)
