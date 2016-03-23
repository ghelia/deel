
from deel import *

CNN = GoogLeNet()

Input("deel.png")
t = CNN.classify()
Show()
print t.value