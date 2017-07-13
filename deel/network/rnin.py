import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph as c
from deel.tensor import *
from deel.network import *
import chainer.serializers as cs
import copy

from deel.deel import *
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
	Network in Network by Chainer 
'''
import deel.model.rnin


class RegionalNetworkInNetwork(ImageNet):
	def __init__(self,modelpath=None,mean='misc/ilsvrc_2012_mean.npy',labels='data/labels.txt',optimizer=None):
		super(RegionalNetworkInNetwork,self).__init__('RegionalNetworkInNetwork',in_size=227)

		self.func = deel.model.rnin.RNIN()
		if modelpath is not None:
			cs.load_hdf5("misc/"+modelpath,self.func)

		self.graph_generated=None

		xp = Deel.xp
		#ImageNet.mean_image = pickle.load(open(mean, 'rb'))
		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=xp.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = self.func.insize

		self.labels = np.loadtxt(labels, str, delimiter="\t")

		self.t = ChainerTensor(Variable(Deel.xp.asarray([1.0])))

		if Deel.gpu>=0:
			self.func.to_gpu()


		if optimizer is None:
			self.optimizer = optimizers.Adam()
		self.optimizer.setup(self.func)


	def forward(self,x):
		y = self.func.forward(x)
		return y

	def classify(self,x=None,train=False):
		if x is None:
			x=Tensor.context

		image = x.value
		self.x_batch = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch,dtype=Deel.xp.float32)
		
		x = chainer.Variable(x_data, volatile='off' if train else 'on')
		score = self.forward(x)
		
		score = F.softmax(score)

		score = Variable(score.data) #Unchain 
		t = ChainerTensor(score)
		t.owner=self
		t.use()
		return t
	def save(self,filename):
		#cs.save_hdf5(filename,self.func.copy().to_cpu())
		cs.save_hdf5(filename,self.func.copy())


	def backprop(self,t,distill=False):
		x=Tensor.context



		self.optimizer.zero_grads()
		if distill:
			t = chainer.Variable(Deel.xp.asarray([t.content.data],dtype=Deel.xp.float32), volatile='off')
			loss = self.func.getLossDistill(x.content,t)
			accuracy = 0.0
		else:
			loss,accuracy = self.func.getLoss(x.content,t.content)
			accuracy = accuracy.data

		loss.backward()
		self.optimizer.update()
		

		if not self.graph_generated:
			#with open('graph.dot', 'w') as o:
			#	o.write(c.build_computational_graph((loss,), False).dump())
			with open('graph.wo_split.dot', 'w') as o:
				o.write(c.build_computational_graph((loss,), True).dump())
			print('generated graph')
			self.graph_generated = True


		return loss.data,accuracy

