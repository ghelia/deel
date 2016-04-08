from deel import *
from deel.network import *
from deel.network.googlenet import *
from deel.commands import *
import cv2

deel = Deel()

CNN = GoogLeNet()
CNN.Input("deel.png")
CNN.classify()
ShowLabels()