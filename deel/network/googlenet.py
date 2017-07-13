import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
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


import deel.model.googlenet as chainermodel



def convert(src):
	dst = chainermodel.GoogLeNet()
	dst.conv1		 = src['conv1/7x7_s2'].copy()
	dst.conv2_reduce = src['conv2/3x3_reduce'].copy()
	dst.conv2        = src['conv2/3x3'].copy()
	dst.inc3a.conv1	 = src['inception_3a/1x1'].copy()
	dst.inc3a.conv3	 = src['inception_3a/3x3'].copy()
	dst.inc3a.conv5	 = src['inception_3a/5x5'].copy()
	dst.inc3a.proj3	 = src['inception_3a/3x3_reduce'].copy()
	dst.inc3a.proj5	 = src['inception_3a/5x5_reduce'].copy()
	dst.inc3a.projp	 = src['inception_3a/pool_proj'].copy()
	dst.inc3b.conv1	 = src['inception_3b/1x1'].copy()
	dst.inc3b.conv3	 = src['inception_3b/3x3'].copy()
	dst.inc3b.conv5	 = src['inception_3b/5x5'].copy()
	dst.inc3b.proj3	 = src['inception_3b/3x3_reduce'].copy()
	dst.inc3b.proj5	 = src['inception_3b/5x5_reduce'].copy()
	dst.inc3b.projp	 = src['inception_3b/pool_proj'].copy()
	dst.inc4a.conv1	 = src['inception_4a/1x1'].copy()
	dst.inc4a.conv3	 = src['inception_4a/3x3'].copy()
	dst.inc4a.conv5	 = src['inception_4a/5x5'].copy()
	dst.inc4a.proj3	 = src['inception_4a/3x3_reduce'].copy()
	dst.inc4a.proj5	 = src['inception_4a/5x5_reduce'].copy()
	dst.inc4a.projp	 = src['inception_4a/pool_proj'].copy()
	dst.inc4b.conv1	 = src['inception_4b/1x1'].copy()
	dst.inc4b.conv3	 = src['inception_4b/3x3'].copy()
	dst.inc4b.conv5	 = src['inception_4b/5x5'].copy()
	dst.inc4b.proj3	 = src['inception_4b/3x3_reduce'].copy()
	dst.inc4b.proj5	 = src['inception_4b/5x5_reduce'].copy()
	dst.inc4b.projp	 = src['inception_4b/pool_proj'].copy()
	dst.inc4c.conv1	 = src['inception_4c/1x1'].copy()
	dst.inc4c.conv3	 = src['inception_4c/3x3'].copy()
	dst.inc4c.conv5	 = src['inception_4c/5x5'].copy()
	dst.inc4c.proj3	 = src['inception_4c/3x3_reduce'].copy()
	dst.inc4c.proj5	 = src['inception_4c/5x5_reduce'].copy()
	dst.inc4c.projp	 = src['inception_4c/pool_proj'].copy()
	dst.inc4d.conv1	 = src['inception_4d/1x1'].copy()
	dst.inc4d.conv3	 = src['inception_4d/3x3'].copy()
	dst.inc4d.conv5	 = src['inception_4d/5x5'].copy()
	dst.inc4d.proj3	 = src['inception_4d/3x3_reduce'].copy()
	dst.inc4d.proj5	 = src['inception_4d/5x5_reduce'].copy()
	dst.inc4d.projp	 = src['inception_4d/pool_proj'].copy()
	dst.inc4e.conv1	 = src['inception_4e/1x1'].copy()
	dst.inc4e.conv3	 = src['inception_4e/3x3'].copy()
	dst.inc4e.conv5	 = src['inception_4e/5x5'].copy()
	dst.inc4e.proj3	 = src['inception_4e/3x3_reduce'].copy()
	dst.inc4e.proj5	 = src['inception_4e/5x5_reduce'].copy()
	dst.inc4e.projp	 = src['inception_4e/pool_proj'].copy()
	dst.inc5a.conv1	 = src['inception_5a/1x1'].copy()
	dst.inc5a.conv3	 = src['inception_5a/3x3'].copy()
	dst.inc5a.conv5	 = src['inception_5a/5x5'].copy()
	dst.inc5a.proj3	 = src['inception_5a/3x3_reduce'].copy()
	dst.inc5a.proj5	 = src['inception_5a/5x5_reduce'].copy()
	dst.inc5a.projp	 = src['inception_5a/pool_proj'].copy()
	dst.inc5b.conv1	 = src['inception_5b/1x1'].copy()
	dst.inc5b.conv3	 = src['inception_5b/3x3'].copy()
	dst.inc5b.conv5	 = src['inception_5b/5x5'].copy()
	dst.inc5b.proj3	 = src['inception_5b/3x3_reduce'].copy()
	dst.inc5b.proj5	 = src['inception_5b/5x5_reduce'].copy()
	dst.inc5b.projp	 = src['inception_5b/pool_proj'].copy()
	dst.loss3_fc     = src['loss3/classifier'].copy()
	dst.loss1_conv   = src['loss1/conv'].copy()
	dst.loss1_fc1    = src['loss1/fc'].copy()
	dst.loss1_fc2    = src['loss1/classifier'].copy()
	dst.loss2_conv   = src['loss2/conv'].copy()
	dst.loss2_fc1    = src['loss2/fc'].copy()
	dst.loss2_fc2    = src['loss2/classifier'].copy()
	return dst

'''
	GoogLeNet by Caffenet 
'''
class GoogLeNet(ImageNet):
	def __init__(self,modelpath='bvlc_googlenet.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='misc/labels.txt',in_size=224):
		super(GoogLeNet,self).__init__('GoogLeNet',in_size)

		if os.path.splitext(modelpath)[1]==".caffemodel":
			self.func = LoadCaffeModel(modelpath)
			self.model = convert(self.func)
		else:
			self.func=None
			self.model = chainermodel.GoogLeNet()
			cs.load_hdf5(modelpath,self.model)

		xp = Deel.xp


		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 103.939
		ImageNet.mean_image[1] = 116.779
		ImageNet.mean_image[2] = 123.68
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
		cs.save_hdf5(filename,self.model.copy().to_cpu())

	def forward(self,x,train=True):
		y = self.model.forward(x)
		return y

	def predict(self, x,train=False):
		if self.func is not None:
			score,= self.func(inputs={'data': x},
					 outputs=['loss3/classifier'], 
					 #disable=['loss1/ave_pool', 'loss2/ave_pool'],
					 train=False)
			return score
		else:
			y = self.model.forward(x)		
		return y[2]


	def classify(self,x=None):
		if x is None:
			x=Tensor.context

		if not isinstance(x,ImageTensor):
			x=Input(x)

		image = x.value
		self.x_batch = image
		xp = Deel.xp
		x_data = xp.asarray(self.x_batch)
		
		x = chainer.Variable(x_data, volatile=True)

		score = self.predict(x)

		score = F.softmax(score)

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

	def feature(self, x,layer=u'loss3/classifier',train=False):
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
		return t

	def batch_feature(self, x):
		if x is None:
			x=Tensor.context

		x = x.content
		self.optimizer.zero_grads()
		outputs = self.forward(x,train=Deel.train)

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

		accuracy = F.accuracy(F.softmax(x.content),t.content)

		if  Deel.train:
			loss.backward()
		self.optimizer.update()
		return loss.data,accuracy.data

