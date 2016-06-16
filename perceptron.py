from deel import *
from deel.network import *
from deel.commands import *
import time
deel = Deel(gpu=-1)

import chainer.functions as F
p = Perceptron(layers=(2,1),activation=F.sigmoid)


x = Tensor(value=[[0,0],[0,1],[1,0],[1,1]])
t = Tensor(value=[[0],[0],[0],[1]])

for i in range(3000):
	print p.forward(x).value
	print p.backprop(t)


start = time.clock()
end = time.clock()
print end -start

