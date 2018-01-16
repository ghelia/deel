	
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph as c
from deel.tensor import *
from deel.network import *
import copy

from deel.deel import *
import deel.network
import chainer
import json
import os
import multiprocessing
import threading
import time
import six
import numpy as np
import os.path
from PIL import Image
from six.moves import queue
import pickle
import hashlib
import datetime
import sys
import random

class AlexNet(ImageNet):
	def __init__(self, model='bvlc_alexnet.caffemodel',mean='misc/ilsvrc_2012_mean.npy',labels='misc/labels.txt',optimizer=None):
		super(AlexNet,self).__init__('AlexNet',in_size=227)


		self.func = LoadCaffeModel(model)
		self.labels = np.loadtxt(labels, str, delimiter="\t")

		if Deel.gpu >= 0:
			cuda.check_cuda_available()


		if Deel.gpu >= 0:
			cuda.get_device(self.gpu).use()
			self.func.to_gpu()

		#ImageNet.mean_image = np.load(mean)
		mean_image = np.load(mean)
		ImageNet.mean_image=mean_image
		
		cropwidth = 256 - self.in_size
		start = cropwidth // 2
		stop = start + self.in_size
		self.mean_image = mean_image[:, start:stop, start:stop].copy()
		#del self.func.layers[15:23] 
		#self.outname = 'pool5'

		self.batchsize = 1
		self.x_batch = np.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

	def forward(self, x,layer='fc8'):
		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)
		return y
				
	def predict(self, x,layer='fc8'):
		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)
		return F.softmax(y)


	def classify(self,x=None):
		if x is None:
			x=Tensor.context

		if not isinstance(x,ImageTensor):
			x=self.Input(x)

		_x = Variable(x.value, volatile=True)
		result = self.predict(_x)
		result = Variable(result.data) #Unchain 
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

	def layerDim(self, layer='fc8'):
		image = self.Input('deel.png').value
		self.x_batch[0] = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)
		x = chainer.Variable(x_data, volatile=True)

		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)

		return y.data.shape
				


	def feature(self, x,layer=u'pool5'):
		if x is None:
			x=Tensor.context
		if not isinstance(x,ImageTensor):
			x=self.Input(x)

		image = x.value

		self.x_batch[0] = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)

		if Deel.gpu >= 0:
			x_data=cuda.to_gpu(x_data)
		
		x = chainer.Variable(x_data, volatile=True)
		score = self.forward(x,layer=layer)

		if Deel.gpu >= 0:
			score=cuda.to_cpu(score.data)
			dim = getDim(score.shape)
			score = score.reshape(dim)
		else:
			dim = getDim(score.data.shape)
			score = score.data.reshape(dim)
		
		score = chainer.Variable(score*255.0, volatile=True)

		t = ChainerTensor(score)
		t.owner=self
		t.use()


		return t
