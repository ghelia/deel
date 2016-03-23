import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers
from chainer.links import caffe
import time
import numpy as np
import os.path
from PIL import Image
import pickle
import cv2
import hashlib

class Tensor():
	""" A tensor """

	context = None

	def __init__(	self,value=np.array([1], dtype=np.float32),
					category='number',comment=''):
		self.content = value

		if category == 'image':
			value = np.asarray(value).transpose(2, 0, 1)
		elif category == 'chainer.Variable':
			value = value.data

		self.shape = value.shape
		self.value = value
		self.comment = comment
		self.category = category

	def use(self):
		Tensor.context = self
	def show(self):
		if self.category == 'image':
			tmp = self.get()
			img = Image.fromarray(tmp)
			img.show()
	def get(self):
		if self.category == 'image':
			return	self.value.transpose(1, 2, 0)
		return self.value



"""Input something to context tensor"""
def Input(x):
	if isinstance(x,str):
		root, ext = os.path.splitext(x)
		if ext=='.png' or ext=='.jpg' or ext=='.jpeg' or ext=='.gif':
			img = Image.open(x)
			t = Tensor(value=img,category='image')
			t.use()
		elif ext=='.txt':
			print "this is txt"

	return t

def CNN_Caffemodel(	model='bvlc_googlenet.caffemodel',
					mean='ilsvrc_2012_mean.npy',
					label='labels.txt'):
	root, ext = os.path.splitext(model)
	cashnpath = 'cash/'+hashlib.sha224(root).hexdigest()+".pkl"
	if os.path.exists(cashnpath):
		func = pickle.load(open(cashnpath,'rb'))
	else:
		func = caffe.CaffeFunction('misc/'+model)
		pickle.dump(func, open(cashnpath, 'wb'))
	mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
	mean_image[0] = 104
	mean_image[1] = 117
	mean_image[2] = 123

	image = Tensor.context.get()

	in_size = 224
	cropwidth = 256 - in_size
	start = cropwidth // 2
	stop = start + in_size
	mean_image = mean_image[:, start:stop, start:stop].copy()
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

	x_batch = np.ndarray(
	        (1, 3, in_size,in_size), dtype=np.float32)
	x_batch[0]=image

	x = Variable(x_batch, volatile=True)
	def predict(x):
		y, = func(inputs={'data': x}, outputs=['loss3/classifier'],
					disable=['loss1/ave_pool', 'loss2/ave_pool'],
					train=False)
		return F.softmax(y)
	t = Tensor(predict(x).data[0])
	t.use()
	return t

