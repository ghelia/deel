import chainer.functions as F
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
#import six.moves.cPickle as pickle
from six.moves import queue
import pickle
import cv2
import hashlib
import sys
import random

class Deel(object):
	singlton = None
	train = None
	val = None
	root = '.'
	epoch=100
	gpu=-1
	def __init__(self):
		self.singleton = self


	@staticmethod
	def getInstance():
		return Deel.singleton

xp = np

def InputBatch(train=None,val=None):
	Deel.train = train
	Deel.val = val

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
	def __init__(	self,x,comment=''):
		super(ImageTensor,self).__init__(
				np.asarray(x).transpose(2, 0, 1),
				comment=comment)
		self.content = x
		
		image = filter(np.asarray(x))

		x_batch = np.ndarray(
				(1, 3, ImageNet.in_size,ImageNet.in_size), dtype=np.float32)
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


"""Input something to context tensor"""
def Input(x):
	if isinstance(x,str):
		root, ext = os.path.splitext(x)
		if ext=='.png' or ext=='.jpg' or ext=='.jpeg' or ext=='.gif':
			img = Image.open(x)
			t = ImageTensor(img)
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
	Trainer
'''
class BatchTrainer(object):
	batchsize=32
	val_batchsize=250
	data_q=None
	res_q=None
	loaderjob=20
	train_list=''
	val_list=''
	def __init__(self,in_size=256):
		BatchTrainer.data_q = queue.Queue(maxsize=1)
		BatchTrainer.res_q = queue.Queue()
		BatchTrainer.in_size=ImageNet.in_size
	def train(self,workout,optimizer=None):
		BatchTrainer.train_list = load(Deel.train,Deel.root)
		BatchTrainer.val_list = load(Deel.val,Deel.root)


		feeder = threading.Thread(target=feed_data)
		feeder.daemon = True
		feeder.start()
		logger = threading.Thread(target=log_result)
		logger.daemon = True
		logger.start()	

		BatchTrainer.workout = workout

		train_loop()
		feeder.join()
		logger.join()



def load(path, root):
	tuples = []
	for line in open(path):
		pair = line.strip().split()
		tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
	return tuples

def feed_data():
	# Data feeder
	i = 0
	count = 0
	in_size = BatchTrainer.in_size
	batchsize = BatchTrainer.batchsize
	val_batchsize = BatchTrainer.val_batchsize
	train_list = BatchTrainer.train_list

	x_batch = np.ndarray(
		(batchsize, 3, in_size, in_size), dtype=np.float32)
	y_batch = np.ndarray((batchsize,), dtype=np.int32)
	val_x_batch = np.ndarray(
		(val_batchsize, 3, in_size, in_size), dtype=np.float32)
	val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)

	batch_pool = [None] * batchsize
	val_batch_pool = [None] * val_batchsize
	pool = multiprocessing.Pool(BatchTrainer.loaderjob)
	BatchTrainer.data_q.put('train')
	for epoch in six.moves.range(1, 1 + Deel.epoch):
		print('epoch', epoch)
		
		perm = np.random.permutation(len(train_list))
		for idx in perm:
			path, label = train_list[idx]
			batch_pool[i] = pool.apply_async(read_image, (path, False, True))
			y_batch[i] = label
			i += 1

			if i == BatchTrainer.batchsize:

				for j, x in enumerate(batch_pool):
					x.wait()
					x_batch[j] = x.get()
				BatchTrainer.data_q.put((x_batch.copy(), y_batch.copy()))
				i = 0

			count += 1
			if count % 100000 == 0:
				BatchTrainer.data_q.put('val')
				j = 0
				for path, label in val_list:
					val_batch_pool[j] = pool.apply_async(
						read_image, (path, True, False))
					val_y_batch[j] = label
					j += 1

					if j == args.val_batchsize:
						for k, x in enumerate(val_batch_pool):
							val_x_batch[k] = x.get()
						BatchTrainer.data_q.put((val_x_batch.copy(), val_y_batch.copy()))
						j = 0
				BatchTrainer.data_q.put('train')

		optimizer.lr *= 0.97
	pool.close()
	pool.join()
	BatchTrainer.data_q.put('end')

def log_result():
	# Logger
	train_count = 0
	train_cur_loss = 0
	train_cur_accuracy = 0
	begin_at = time.time()
	val_begin_at = None
	while True:
		result = BatchTrainer.res_q.get()
		if result == 'end':
			break
		elif result == 'train':
			train = True
			if val_begin_at is not None:
				begin_at += time.time() - val_begin_at
				val_begin_at = None
			continue
		elif result == 'val':
			train = False
			val_count = val_loss = val_accuracy = 0
			val_begin_at = time.time()
			continue

		loss, accuracy = result
		if train:
			train_count += 1
			duration = time.time() - begin_at
			throughput = train_count * args.batchsize / duration
			print(
				'\rtrain {} updates ({} samples) time: {} ({} images/sec)'
				.format(train_count, train_count * args.batchsize,
						datetime.timedelta(seconds=duration), throughput))

			train_cur_loss += loss
			train_cur_accuracy += accuracy
			if train_count % 1000 == 0:
				mean_loss = train_cur_loss / 1000
				mean_error = 1 - train_cur_accuracy / 1000
				print(json.dumps({'type': 'train', 'iteration': train_count,
								  'error': mean_error, 'loss': mean_loss}))
				sys.stdout.flush()
				train_cur_loss = 0
				train_cur_accuracy = 0
		else:
			val_count += args.val_batchsize
			duration = time.time() - val_begin_at
			throughput = val_count / duration
			print(
				'\rval   {} batches ({} samples) time: {} ({} images/sec)'
				.format(val_count / args.val_batchsize, val_count,
						datetime.timedelta(seconds=duration), throughput))

			val_loss += loss
			val_accuracy += accuracy
			if val_count == 50000:
				mean_loss = val_loss * args.val_batchsize / 50000
				mean_error = 1 - val_accuracy * args.val_batchsize / 50000
				print(json.dumps({'type': 'val', 'iteration': train_count,
								  'error': mean_error, 'loss': mean_loss}))


def train_loop():
	global workout
	while True:
		while BatchTrainer.data_q.empty():
			time.sleep(0.1)
		inp = BatchTrainer.data_q.get()
		print "-----"
		print inp
		if inp == 'end':  # quit
			BatchTrainer.res_q.put('end')
			break
		elif inp == 'train':  # restart training
			BatchTrainer.res_q.put('train')
			volatile = 'off'
			continue
		elif inp == 'val':  # start validation
			BatchTrainer.res_q.put('val')
			volatile = 'on'
			continue

		x = ChainerTensor(Variable(xp.asarray(inp[0]), volatile=volatile))
		t = ChainerTensor(Variable(xp.asarray(inp[1]), volatile=volatile))

		loss,accuracy = workout(x,t)
		

		BatchTrainer.res_q.put((float(loss.data), float(accuracy.data)))
		del x, t



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

def read_image(path, center=False, flip=False):
	cropwidth = 256 - ImageNet.in_size
	image = Image.open(path)
	image = np.asarray(image)
	#resizing
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

	image = np.asarray(image).transpose(2, 0, 1)

	if center:
		top = left = cropwidth / 2
	else:
		top = random.randint(0, cropwidth - 1)
		left = random.randint(0, cropwidth - 1)
	bottom = ImageNet.in_size + top
	right = ImageNet.in_size + left

	image = image[:, top:bottom, left:right].astype(np.float32)
	image -= ImageNet.mean_image[:, top:bottom, left:right]
	image /= 255
	if flip and random.randint(0, 1) == 0:
		return image[:, :, ::-1]
	else:
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

		x = Variable(x.value, volatile=True)
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
	def __init__(self,
					labels='labels.txt',optimizer=None):
		super(NetworkInNetwork,self).__init__('NetworkInNetwork',in_size=227)

		self.func = model.nin.NIN()


		ImageNet.mean_image = np.ndarray((3, 256, 256), dtype=np.float32)
		ImageNet.mean_image[0] = 104
		ImageNet.mean_image[1] = 117
		ImageNet.mean_image[2] = 123

		self.labels = np.loadtxt("misc/"+labels, str, delimiter="\t")


		if optimizer == None:
			self.optimizer = optimizers.MomentumSGD(lr=0.01, momentum=0.9)
		self.optimizer.setup(self.func)


	def forward(self,x):
		y = self.func.forward()
		return y

	def classify(self,x=None):
		if x==None:
			x=Tensor.context

		x = Variable(x.value, volatile=True)
		result = self.forward(x)
		t = ChainerTensor(result)
		t.owner=self
		t.use()

		return t

	def backprop(self,x=None,t=None):
		if x==None:
			x=Tensor.context

		loss,accuracy = self.func.loss(result,x.content,t.content)
		t.loss =loss
		t.accuracy=accuracy
		loss.backward()
		self.optimizer.update()
		print('learning rate', optimizer.lr)

		return t


def BatchTrain(callback):
	global workout
	trainer = BatchTrainer()
#	trainer.train(workout)

	BatchTrainer.train_list = load(Deel.train,Deel.root)
	BatchTrainer.val_list = load(Deel.val,Deel.root)


	feeder = threading.Thread(target=feed_data)
	feeder.daemon = True
	feeder.start()
	logger = threading.Thread(target=log_result)
	logger.daemon = True
	logger.start()	

	workout = callback

	train_loop()
	feeder.join()
	logger.join()



def Show(x=None):
	if x==None:
		x = Tensor.context

	x.show()

def ShowLabels(x=None):
	if x==None:
		x = Tensor.context

	t = LabelTensor(x)

	t.show()


def Output(x=None,num_of_candidate=5):
	if x==None:
		x = Tensor.context

	out = x.output

	out.sort(cmp=lambda a, b: cmp(a[0], b[0]), reverse=True)
	for rank, (score, name) in enumerate(out[:num_of_candidate], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))


