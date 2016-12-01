from deel import *
from deel.network import *
from deel.network.caffenet import *
from deel.commands import *
from deel.network.googlenet import *
import chainer.functions as F
import time

deel = Deel()

CNN = CaffeNet(modelpath='EmotiW_VGG_S.caffemodel',in_size=228,
			outputLayers=['fc8'],
			labels= ['Angry' , 'Disgust' , 'Fear' , 'Happy' , 'Fear' , 'Sad' , 'Surprise']
			)

import cv2
cam = cv2.VideoCapture(0)  
cnt=0
print "start"
import time
while True:
	ret, img = cam.read()  
	CNN.Input(img)
	t = CNN.classify()
	#time.sleep(10)
	ShowLabels()
	if t.value[0][2]<0.5:
		print "something happen ",int(t.value[0][2]*10)
		cv2.imwrite('img/face/{}.png'.format(cnt),img)
		cnt+=1

	cv2.imshow('cam', img)
	if cv2.waitKey(10) > 0:
		break
cam.release()
cv2.destroyAllWindows()