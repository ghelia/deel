import cv2 
from deel import *
from deel.network import *
from deel.commands import *

deel = Deel()

CNN = GoogLeNet()

cam = cv2.VideoCapture(0)  

while True:
	ret, img = cam.read()  
	CNN.Input(img)
	CNN.classify()

	labels = GetLabels()
	if labels[0][1] == 'Band':
		print 'BAND'
		cv2.imwrite('band.png',img)

	cv2.imshow('cam', img)
	#if cv2.waitKey(10) > 0:
	#	break
cam.release()
cv2.destroyAllWindows()


"""
InputBatch(train="data/train_lstm.tsv")

CNN = GoogLeNet()
RNN = LSTM()

def trainer(x,t):
	CNN.classify(x) 
	RNN.learn(t)
	return RNN.backprop()

BatchTrain(trainer)
"""
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
"""
