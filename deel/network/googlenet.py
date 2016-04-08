import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph as c
from deel.tensor import *
from deel.network import *
import copy

from deel import *
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


'''
	GoogLeNet by Caffenet 
'''
class GoogLeNet(ImageNet):
	def __init__(self,modelpath='bvlc_googlenet.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='labels.txt',in_size=224):
		super(GoogLeNet,self).__init__('GoogLeNet',in_size)

		self.func = LoadCaffeModel(modelpath)

		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = in_size

		self.labels = np.loadtxt("misc/"+labels, str, delimiter="\t")
		self.batchsize = 1
		self.x_batch = np.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

	def forward(self,x):
		y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
					disable=['loss1/ave_pool', 'loss2/ave_pool'],
					train=False)
		return F.softmax(y)

	def predict(self, x,layer='fc8'):
		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)
		return F.softmax(y)


	def classify(self,x=None):
		if x is None:
			x=Tensor.context

		_x = Variable(x.value, volatile=True)
		result = self.forward(_x)
		result = Variable(result.data) #Unchain 
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

	def layerDim(self, layer='inception_5b/pool_proj'):
		image = self.Input('deel.png').value
		self.x_batch[0] = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)
		x = chainer.Variable(x_data, volatile=True)

		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)

		return y.data.shape

	def feature(self, x,layer=u'inception_5b/pool_proj'):
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
		score = self.predict(x,layer=layer)

		if Deel.gpu >= 0:
			score=cuda.to_cpu(score.data)
			dim = getDim(score.shape)
			score = score.reshape(dim)
		else:
			dim = getDim(score.data.shape)
			score = score.data.reshape(dim)
		
		score = chainer.Variable(score, volatile=True)

		t = ChainerTensor(score)
		t.owner=self
		t.use()


		return t
