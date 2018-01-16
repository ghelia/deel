import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
import deel.model.caffefunction as CaffeFunction
from chainer import computational_graph as c
import chainer.serializers as cs
from deel.tensor import *
from deel.network import *
import copy

from deel.deel import *
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

	Please download .caffemodel and put on /misc

	ResNet-152-model.caffemodel 
	One Drive download https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777

	https://github.com/KaimingHe/deep-residual-networks	

'''
class ResNet152(ImageNet):
	def __init__(self,modelpath='ResNet-152-model.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='misc/labels.txt',in_size=224,
					tuning_layer='fc1000'):
		super(ResNet152,self).__init__('ResNet152',in_size)

		if os.path.splitext(modelpath)[1]==".caffemodel":
			self.func = LoadCaffeModel(modelpath)
		else:
			self.func = LoadCaffeModel("ResNet-152-model.caffemodel")
			cs.load_hdf5(modelpath,self.func)

		xp = Deel.xp


		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = in_size

		self.labels = np.loadtxt(labels, str, delimiter="\t")
		self.batchsize = 1
		self.x_batch = xp.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

		if Deel.gpu >=0:
			self.func = self.func.to_gpu(Deel.gpu)
		#self.optimizer = optimizers.MomentumSGD(lr=0.01,momentum=0.9)
		self.optimizer = optimizers.RMSpropGraves()
		#self.optimizer.setup(self.func.fc1000)
		self.optimizer.setup(self.func[tuning_layer])
	def save(self,filename):
		cs.save_hdf5(filename,self.func.to_cpu())

	def forward(self,x,train=True):
		y, = self.func(inputs={'data': x}, outputs=['fc1000'],
			train=train)
		return y

	def predict(self, x,train=False):
		return F.softmax(self.forward(x,train=train))


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


	def batch_feature(self, x,t):
		if x is None:
			x=Tensor.context

		x = x.content
		self.optimizer.zero_grads()
		y = self.forward(x,train=True)

		t = ChainerTensor(y)
		t.owner=self
		t.use()

		return t

	def backprop(self,t,x=None):
		if x is None:
			x=Tensor.context
		loss = F.softmax_cross_entropy(x.content,t.content)

		accuracy = F.accuracy(F.softmax(x.content),t.content)

		if  Deel.train:
			loss.backward()
			self.optimizer.update()
		return loss.data,accuracy.data

