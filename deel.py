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
		self.output = None
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
	def __init__(self,name):
		super(ImageNet,self).__init__(name)

	def _filter(self,image):
		cropwidth = 256 - self.in_size
		start = cropwidth // 2
		stop = start + self.in_size
		mean_image = self.mean_image[:, start:stop, start:stop].copy()
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
					labels='labels.txt'):
		super(GoogLeNet,self).__init__('GoogLeNet')

		root, ext = os.path.splitext(model)
		cashnpath = 'cash/'+hashlib.sha224(root).hexdigest()+".pkl"
		if os.path.exists(cashnpath):
			self.func = pickle.load(open(cashnpath,'rb'))
		else:
			self.func = caffe.CaffeFunction('misc/'+model)
			pickle.dump(func, open(cashnpath, 'wb'))
		self.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		self.mean_image[0] = 104
		self.mean_image[1] = 117
		self.mean_image[2] = 123

		self.labels = np.loadtxt("misc/"+labels, str, delimiter="\t")

	def predict(self,x):
		y, = self.func(inputs={'data': x}, outputs=['loss3/classifier'],
					disable=['loss1/ave_pool', 'loss2/ave_pool'],
					train=False)
		return F.softmax(y)

	def classify(self,x=None):
		if x==None:
			x=Tensor.context.get()

		image = x

		self.in_size = 224
		image = super(GoogLeNet,self)._filter(image)

		x_batch = np.ndarray(
		        (1, 3, self.in_size,self.in_size), dtype=np.float32)
		x_batch[0]=image

		x = Variable(x_batch, volatile=True)
		t = Tensor(self.predict(x).data[0])
		t.use()
		t.output=zip(t.value.tolist(), self.labels)

		return t


def Output(x=None,num_of_candidate=5):
	if x==None:
		x = Tensor.context

	out = x.output

	out.sort(cmp=lambda a, b: cmp(a[0], b[0]), reverse=True)
	for rank, (score, name) in enumerate(out[:num_of_candidate], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))


