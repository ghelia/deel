from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *
import cv2

deel = Deel()

CNN = GoogLeNet()

cam = cv2.VideoCapture(0)  

while True:
	ret, img = cam.read()  
	CNN.Input(img)
	CNN.classify()
	ShowLabels()

	#cv2.imshow('cam', img)
	#if cv2.waitKey(10) > 0:
	#	break
cam.release()
cv2.destroyAllWindows()
