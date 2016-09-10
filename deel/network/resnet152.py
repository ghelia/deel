import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph as c
import chainer.serializers as cs
from deel.tensor import *
from deel.network import *
import copy

from deel import *
import deel.network
import chainer
import json
import os
import os.path
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
	ResNet152 by Caffenet ModelZoo
'''
class ResNet152(ImageNet):
	def __init__(self,modelpath='ResNet-152-model.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='misc/labels.txt',in_size=224):
		super(ResNet152,self).__init__('ResNet152',in_size)

		if os.path.splitext(modelpath)[1]==".caffemodel":
			self.func = LoadCaffeModel(modelpath)
			self.model = self.func.copy()
		else:
			self.model = chainermodel.GoogLeNet()
			cs.load_hdf5(modelpath,self.model)

		xp = Deel.xp


		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = in_size

		#print type(ImageNet.mean_image)
		self.labels = np.loadtxt(labels, str, delimiter="\t")
		self.batchsize = 1
		self.x_batch = xp.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

		if Deel.gpu >=0:
			self.model = self.model.to_gpu(Deel.gpu)
		self.optimizer = optimizers.MomentumSGD(lr=0.01,momentum=0.9)
		#self.optimizer = optimizers.Adam()
		#self.optimizer.setup(self.func)
		self.optimizer.setup(self.model)
	def save(self,filename):
		cs.save_hdf5(filename,self.model.to_cpu())

	def forward(self,x,train=True):
		y = self.model(x)
		return y

	def predict(self, x,train=False):
		y, = self.func(inputs={'data': x}, outputs=['fc1000'],
			train=train)
		return F.softmax(y)


	def classify(self,x=None):
		if x is None:
			x=Tensor.context

		if not isinstance(x,ImageTensor):
			x=Input(x)

		image = x.value
		self.x_batch = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)
		x = chainer.Variable(x_data, volatile='on')
		score = self.predict(x)

		score = Variable(score.data) #Unchain 
		t = ChainerTensor(score)
		t.owner=self
		t.use()

		return t

	def layerDim(self, layer='inception_5b/pool_proj'):
		xp = Deel.xp
		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=xp.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123

		image = self.Input('deel.png').value
		self.x_batch[0] = image
		x_data = xp.asarray(self.x_batch)
		x = chainer.Variable(x_data, volatile=True)

		y, = self.func(inputs={'data': x}, outputs=[layer], train=False)

		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
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
		score, = self.func(inputs={'data': x}, outputs=[layer],
			disable=['loss1/ave_pool', 'loss2/ave_pool'],
			train=train)


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

	def batch_feature(self, x,t):
		if x is None:
			x=Tensor.context

		x = x.content
		self.optimizer.zero_grads()
		outputs = self.forward(x,train=True)

		t = ChainerTensor(outputs[2])
		t.owner=self
		t.use()

		self.loss1=outputs[0]
		self.loss2=outputs[1]
		self.loss3=outputs[2]


		return t

	def backprop(self,t,x=None):
		if x is None:
			x=Tensor.context
		loss1 = F.softmax_cross_entropy(self.loss1,t.content)
		loss2 = F.softmax_cross_entropy(self.loss2,t.content)
		loss3 = F.softmax_cross_entropy(self.loss3,t.content)

		loss = 0.3*(loss1+loss2) +loss3

		accuracy = F.accuracy(x.content,t.content)

		if  Deel.train:
			loss.backward()
		self.optimizer.update()
		return loss.data,accuracy.data

