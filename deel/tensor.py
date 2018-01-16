from chainer import Variable, FunctionSet, optimizers
from chainer.links import caffe
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
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
from .deel import *
from functools import cmp_to_key

class Tensor(object):
	""" A tensor """

	context = None

	def __init__(	self,value=np.array([1], dtype=np.float32),
					category='scalar',comment=''):
		if type(value)=='numpy.ndarray':
			self.content = np.array(value,dtype=np.float32)
		else:
			self.content = value
		self.shape = self.content.shape
		self.value = self.content 
		self.comment = comment
		self.output = None
		self.owner = None
	def use(self):
		Tensor.context = self
	def show(self):
			print(self.get())
	def get(self):
		return self.value



class ImageTensor(Tensor):
	def __init__(	self,x,filtered_image=None,in_size=256,h=None,w=None,comment='',path=None):
		super(ImageTensor,self).__init__(
				np.asarray(x).transpose(2, 0, 1),
				comment=comment)
		self.content = x
		self.path=path

		
		if filtered_image is None:
			filtered_image=_x
		image = filtered_image
		

		xp = Deel.xp
		if w is None:
			w = in_size
		if h is None:
			h = in_size

		x_batch = xp.ndarray(
				(1, 3, h,w), dtype=xp.float32)

		x_batch[0]=xp.asarray(image)

		self.value=x_batch

	def get(self):
		return	self.value.transpose(1, 2, 0)
	def show(self):
		tmp = self.get()
		img = Image.fromarray(tmp)
		img.show()

	def __del__(self):
		del self.value
		del self.content

class ChainerTensor(Tensor):
	def __init__(	self,x,comment=''):
		super(ChainerTensor,self).__init__(
				x.data,
				comment=comment)
		self.content = x
		self.value = x.data
	def __del__(self):
		del self.value
		del self.content

class LabelTensor(Tensor):
	def __init__(	self,x,comment=''):
		super(LabelTensor,self).__init__(
				x,
				comment=comment)
		out=list(zip(x.value[0].tolist(), x.owner.labels))
		out.sort(key=cmp_to_key(lambda a, b: ((a[0] > b[0]) - (a[0] < b[0]))), reverse=True)
		self.content = out

	def show(self,num_of_candidate=20):
		for rank, (score, name) in enumerate(self.content[:num_of_candidate], start=1):
			print('#%d | %s | %4.1f%%' % (rank, name, score * 100))	

