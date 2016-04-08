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
from deel import *

class Tensor(object):
	""" A tensor """

	context = None

	def __init__(	self,value=np.array([1], dtype=np.float32),
					category='scalar',comment=''):
		self.content = value
		self.shape = value.shape
		self.value = value
		self.comment = comment
		self.output = None
		self.owner = None
	def use(self):
		Tensor.context = self
	def show(self):
			print self.get()
	def get(self):
		return self.value



class ImageTensor(Tensor):
	def __init__(	self,x,filtered_image=None,in_size=256,comment=''):
		super(ImageTensor,self).__init__(
				np.asarray(x).transpose(2, 0, 1),
				comment=comment)
		self.content = x
		
		if filtered_image is None:
			filtered_image=_x
		image = filtered_image
		

		x_batch = np.ndarray(
				(1, 3, in_size,in_size), dtype=np.float32)
		x_batch[0]=image

		self.value=x_batch

	def get(self):
		return	self.value.transpose(1, 2, 0)
	def show(self):
		tmp = self.get()
		img = Image.fromarray(tmp)
		img.show()


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
		out=zip(x.value[0].tolist(), x.owner.labels)
		out.sort(cmp=lambda a, b: cmp(a[0], b[0]), reverse=True)
		self.content = out

	def show(self,num_of_candidate=5):
		for rank, (score, name) in enumerate(self.content[:num_of_candidate], start=1):
			print('#%d | %s | %4.1f%%' % (rank, name, score * 100))	

