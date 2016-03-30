import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph
from tensor import *
from network import *
from deel import *
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
	ImageNet
'''
class ImageNet(Network):
	mean_image=None
	in_size=224
	mean_image=None
	def __init__(self,name,in_size=224):
		super(ImageNet,self).__init__(name)
		ImageNet.in_size = in_size
	def Input(self,path):
		img = Image.open(path)
		print path 
		print self.in_size
		t = ImageTensor(img,filtered_image=filter(np.asarray(img)),
						in_size=self.in_size)
		t.use()



	
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
			pickle.dump(this.func, open(cashnpath, 'wb'))
		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123
		ImageNet.in_size = in_size

		self.labels = np.loadtxt("misc/"+labels, str, delimiter="\t")

	def forward(self,x):
		y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
					disable=['loss1/ave_pool', 'loss2/ave_pool'],
					train=False)
		return F.softmax(y)

	def classify(self,x=None):
		if x==None:
			x=Tensor.context

		_x = Variable(x.value, volatile=True)
		result = self.forward(_x)
		result = Variable(result.data) #Unchain 
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
		ImageNet.in_size = model.nin.insize

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

		_x = x.content
		result = self.forward(_x)
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

	def train(self,x,t):
		if x==None:
			x=Tensor.context
		_x = x.content
		_t = t.content
		loss= self.func(_x,_t)
		loss.backward()
		self.optimizer.update()
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


import model.lstm
class LSTM(Network):
	def __init__(self,optimizer=None,vocab=None,n_input_units=1000,
					n_units=650,grad_clip=5,bproplen=35):

		if vocab==None:
			vocab=BatchTrainer.vocab
		self.vocab=vocab
		n_vocab = len(vocab)
		super(LSTM,self).__init__('LSTM')

		self.func = model.lstm.RNNLM(n_input_units=n_input_units,n_vocab=n_vocab,n_units=n_units)
		self.func.compute_accuracy = False 
		for param in self.func.params():
			data = param.data
			data[:] = np.random.uniform(-0.1, 0.1, data.shape)


		if Deel.gpu>=0:
			self.func.to_gpu()


		if optimizer == None:
			self.optimizer = optimizers.SGD(lr=1.)
		self.optimizer.setup(self.func)
		self.clip = chainer.optimizer.GradientClipping(grad_clip)
		self.optimizer.add_hook(self.clip)

		self.accum_loss = 0
		self.cur_log_perp =  Deel.xp.zeros(())

	def evaluate(dataset):
		# Evaluation routine
		evaluator = self.func.copy()  # to use different state
		evaluator.predictor.reset_state()  # initialize state
		evaluator.predictor.train = False  # dropout does nothing

		sum_log_perp = 0
		for i in six.moves.range(dataset.size - 1):
			x = chainer.Variable(Deel.xp.asarray(dataset[i:i + 1]), volatile='on')
			t = chainer.Variable(Deel.xp.asarray(dataset[i + 1:i + 2]), volatile='on')
			loss = evaluator(x, t)
			sum_log_perp += loss.data
		return math.exp(float(sum_log_perp) / (dataset.size - 1))

	def forward(self,x=None):
		return self.func(x)

	def learn(self,str,x=None):
		if x==None:
			x=Tensor.context

		_t = Deel.xp.asarray([self.vocab[str[0]]], dtype=np.int32)
		t = ChainerTensor(Variable(_t))
		self.firstInput(t)
		#_x = x.content



		for j in range(len(str)-2):
			_x = Deel.xp.asarray([self.vocab[str[j+1]]], dtype=np.int32)
			x = ChainerTensor(Variable(_x))
			x.use()

			_t = Deel.xp.asarray([self.vocab[str[j+2]]], dtype=np.int32)
			t = ChainerTensor(Variable(_t))
			self.train(t)

		print ('loss',self.accum_loss.data)

		return x



	def firstInput(self,t,x=None):
		if x==None:
			x=Tensor.context
		_x = x.content
		_t = t.content
		_y = self.func(_x,mode=1)
		loss = chainer.functions.loss.softmax_cross_entropy.softmax_cross_entropy(_y,_t)
		self.func.y = _y
		self.func.loss = loss
		self.accum_loss += loss
		self.cur_log_perp += loss.data

		return x

	def train(self,t,x=None):
		if x==None:
			x=Tensor.context
		_x = x.content
		_t = t.content

		_y = self.func(_x)
		loss= F.softmax_cross_entropy(_y,_t)
		self.func.y = _y
		self.func.loss = loss
		self.accum_loss += loss
		self.cur_log_perp += loss.data

		return self


	def backprop(self):
		self.func.zerograds()
		self.accum_loss.backward()
		self.accum_loss.unchain_backward()  # truncate
		self.accum_loss = 0
		self.optimizer.update()
		
import collections
def _sum_sqnorm(arr):
    sq_sum = collections.defaultdict(float)
    for x in arr:
        with cuda.get_device(x) as dev:
            x = x.ravel()
            s = x.dot(x)
            sq_sum[int(dev)] += s
    return sum([float(i) for i in six.itervalues(sq_sum)])

