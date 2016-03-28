import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers
from chainer.links import caffe
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from tensor import *
from network import *
from deel import *
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
import cv2
import hashlib
import datetime
import sys
import random


class Network(object):
	def __init__(self,name):
		self.name=name
		self.func=None
	def predict(self,x):
		'''
			Forward neural Network to prediction
		'''
		return None
	def classify(self,x=None):
		'''
			Classify x
		'''
		return None
	def trainer(self):
		'''
			Trainer for neural network
		'''
		return None
	def __str__(self):
		return self.name

'''
	ImageNet
'''
class ImageNet(Network):
	mean_image=None
	in_size=224
	mean_image=None
	def __init__(self,name,in_size=224):
		super(ImageNet,self).__init__(name)
		ImageNet.in_size = in_size

def filter(image):
	cropwidth = 256 - ImageNet.in_size
	start = cropwidth // 2
	stop = start + ImageNet.in_size
	mean_image = ImageNet.mean_image[:, start:stop, start:stop].copy()
	target_shape = (256, 256)
	output_side_length=256

	height, width, depth = image.shape
	new_height = output_side_length
	new_width = output_side_length
	if height > width:
		new_height = output_side_length * height / width
	else:
		new_width = output_side_length * width / height
	resized_img = cv2.resize(image, (new_width, new_height))
	height_offset = (new_height - output_side_length) / 2
	width_offset = (new_width - output_side_length) / 2
	image= resized_img[height_offset:height_offset + output_side_length,
	width_offset:width_offset + output_side_length]

	image = image.transpose(2, 0, 1)
	image = image[:, start:stop, start:stop].astype(np.float32)
	image -= mean_image

	return image


	
'''
	GoogLeNet by Caffenet 
'''
class GoogLeNet(ImageNet):
	def __init__(self,model='bvlc_googlenet.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					labels='labels.txt',in_size=224):
		super(GoogLeNet,self).__init__('GoogLeNet',in_size)

		root, ext = os.path.splitext(model)
		cashnpath = 'cash/'+hashlib.sha224(root).hexdigest()+".pkl"
		if os.path.exists(cashnpath):
			self.func = pickle.load(open(cashnpath,'rb'))
		else:
			self.func = caffe.CaffeFunction('misc/'+model)
			pickle.dump(func, open(cashnpath, 'wb'))
		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123

		self.labels = np.loadtxt("misc/"+labels, str, delimiter="\t")

	def forward(self,x):
		y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
					disable=['loss1/ave_pool', 'loss2/ave_pool'],
					train=False)
		return F.softmax(y)

	def classify(self,x=None):
		if x==None:
			x=Tensor.context

		x = x.content
		result = self.forward(x)
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

'''
	Network in Network by Chainer 
'''
import model.nin

class NetworkInNetwork(ImageNet):
	def __init__(self,mean='data/mean.npy',labels='data/labels.txt',optimizer=None):
		super(NetworkInNetwork,self).__init__('NetworkInNetwork',in_size=227)

		self.func = model.nin.NIN()

		ImageNet.mean_image = pickle.load(open(mean, 'rb'))

		self.labels = np.loadtxt(labels, str, delimiter="\t")

		if Deel.gpu>=0:
			self.func.to_gpu()


		if optimizer == None:
			self.optimizer = optimizers.MomentumSGD(Deel.optimizer_lr, momentum=0.9)
		self.optimizer.setup(self.func)


	def forward(self,x):
		y = self.func.forward(x)
		return y

	def classify(self,x=None):
		if x==None:
			x=Tensor.context

		x = x.content
		result = self.forward(x)
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

	def train(self,x,t):
		_x = x.content
		_t = t.content
		loss= self.func(_x,_t)
		print("backward")
		loss.backward()
		print("backward-end")
		self.optimizer.update()
		print('loss', loss.data)
		t.content.loss =loss
		t.content.accuracy=self.func.accuracy		
		return t


	def backprop(self,t):
		x=Tensor.context


		self.optimizer.lr = Deel.optimizer_lr

		self.optimizer.zero_grads()
		loss,accuracy = self.func.getLoss(x.content,t.content)
		t.content.loss =loss
		t.content.accuracy=accuracy
		loss.backward()
		self.optimizer.update()

		return t
