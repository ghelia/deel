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
	CaffeNet Infference
'''
class CaffeNet(ImageNet):
	def __init__(self,modelpath='gender_net.caffemodel.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='misc/labels.txt',in_size=228,
					disableLayers=[],
					outputLayers=['fc8']
					):
		super(CaffeNet,self).__init__('CaffeNet',in_size)

		self.func = LoadCaffeModel(modelpath)

		xp = Deel.xp
		self.disableLayers = disableLayers
		self.outputLayers = outputLayers


		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = in_size

		if type(labels) is str:
			self.labels = np.loadtxt(labels, str, delimiter="\t")
		else:
			self.labels=labels
		self.batchsize = 1
		self.x_batch = xp.ndarray((self.batchsize, 3, self.in_size, self.in_size), dtype=np.float32)

		if Deel.gpu >=0:
			self.func = self.func.to_gpu(Deel.gpu)


	def save(self,filename):
		cs.save_hdf5(filename,self.func.to_cpu())

	def forward(self,x):
		y, = self.func(inputs={'data': x}, disable=self.disableLayers,outputs=self.outputLayers,
			train=False)
		return y

	def predict(self, x):
		return F.softmax(self.forward(x))


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


