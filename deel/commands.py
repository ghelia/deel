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
#import six.moves.cPickle as pickle
from six.moves import queue
import pickle
import hashlib
import datetime
import sys
import random
import cv2


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




def load(path, root):
	tuples = []
	for line in open(path):
		pair = line.strip().split()
		tuples.append((os.path.join(root, pair[0]), np.int32(pair[1])))
	return tuples


def read_image(path, center=False, flip=False):
	cropwidth = 256 - ImageNet.in_size
	try:
		image = np.asarray(Image.open(path)).transpose(2, 0, 1)
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
	except ValueError:
		print "---- Read_Image Value Error ----"
		print path

		return None


def batch_read_and_feed(batch):
	batchsize = BatchTrainer.batchsize
	in_size = BatchTrainer.in_size
	x_batch = np.ndarray(
		(batchsize, 3, in_size, in_size), dtype=np.float32)
	y_batch = np.ndarray((batchsize,), dtype=np.int32)
	i=0
	for data in batch:
		path,label = data
		#img = read_image (path, False, True)
		img = filter(np.asarray(cv2.imread(path,cv2.IMREAD_COLOR)),flip=True,center=False)
		if img is not None:
			x_batch[i] = img
			y_batch[i] = label
			i+=1

	return x_batch,y_batch


def feed_data():
	global optimizer_lr,checkout
	# Data feeder
	i = 0
	count = 0
	in_size = BatchTrainer.in_size
	batchsize = BatchTrainer.batchsize
	val_batchsize = BatchTrainer.val_batchsize
	train_list = BatchTrainer.train_list
	val_list = BatchTrainer.val_list


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

	thleads = 1
	for epoch in six.moves.range(1, 1 + Deel.epoch):
		print('epoch', epoch)
		print('learning rate=%f'%Deel.optimizer_lr)
		
		#perm = np.random.permutation(len(train_list))
		random.shuffle(train_list)

		for idx in range(0,len(train_list)-batchsize*thleads,batchsize*thleads):
			#paths, labels = train_list[idx:idx+batchsize]
			#print paths

			r=[pool.apply_async(batch_read_and_feed, 
					(train_list[idx+i*batchsize:
								idx+i*batchsize+batchsize],)) 
					for i in range(thleads)]
			for res in r:
				x_batch,y_batch = res.get()
				BatchTrainer.data_q.put((x_batch, y_batch))

			count += 1
			if count % 10000 == 0:
				print "checkout"
				checkout()
				BatchTrainer.data_q.put('val')
				j = 0
				#val_list=train_list[idx:idx+val_batchsize]
				#random.shuffle(val_list)
				"""for path, label in val_list:
					val_batch_pool[j] = pool.apply_async(
						read_image, (path, True, False))
					val_y_batch[j] = label
					j += 1

					if j == val_batchsize:
						for k, x in enumerate(val_batch_pool):
							val_x_batch[k] = x.get()
						BatchTrainer.data_q.put((val_x_batch.copy(), val_y_batch.copy()))
						j = 0
				"""
				r=[pool.apply_async(batch_read_and_feed, 
					(val_list[i*val_batchsize:i*val_batchsize+val_batchsize],)) 
					for i in range(len(val_list)/val_batchsize)]				
				for res in r:
					val_x_batch,val_y_batch = res.get()
					BatchTrainer.data_q.put((val_x_batch.copy(), val_y_batch.copy()))
				
				BatchTrainer.data_q.put('train')
		Deel.optimizer_lr *= 0.98

	pool.close()
	pool.join()
	BatchTrainer.data_q.put('end')

def log_result():
	# Logger
	train_count = 0
	train_cur_loss = 0
	train_cur_accuracy = 0
	begin_at = time.time()
	val_batchsize = BatchTrainer.val_batchsize
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
			throughput = train_count * BatchTrainer.batchsize / duration
			if train_count % 100 == 0:
				print(
					'\rtrain {} updates ({} samples) time: {} ({} images/sec)'
					.format(train_count, train_count * BatchTrainer.batchsize,
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
			val_count += val_batchsize
			print "valc",val_count
			duration = time.time() - val_begin_at
			throughput = val_count / duration
			print(
				'\rval   {} batches ({} samples) time: {} ({} images/sec)'
				.format(val_count / val_batchsize, val_count,
						datetime.timedelta(seconds=duration), throughput))

			val_loss += loss
			val_accuracy += accuracy
			if val_count % 1024 ==0:
				mean_loss = val_loss *val_batchsize / 1024
				mean_error = 1 - val_accuracy  * val_batchsize/ 1024
				val_loss=0
				val_accuracy=0
				print(json.dumps({'type': 'val', 'iteration': train_count,
								  'error': mean_error, 'loss': mean_loss}))
import cupy.cuda.runtime as rt
def train_loop():
	global workout
	train=True
	Deel.trainCount=0
	while True:
		while BatchTrainer.data_q.empty():
			time.sleep(0.1)
		inp = BatchTrainer.data_q.get()
		if inp == 'end':  # quit
			BatchTrainer.res_q.put('end')
			break
		elif inp == 'train':  # restart training
			BatchTrainer.res_q.put('train')
			train=True
			continue
		elif inp == 'val':  # start validation
			BatchTrainer.res_q.put('val')
			train=False
			continue
		#train=False

		Deel.train = train

		volatile = 'off' if train else 'on'
		#volatile='on'

		_ax = Deel.xp.asarray(inp[0])
		_at = Deel.xp.asarray(inp[1])

		if Deel.defferedTuning:
			_x =Variable(_ax, volatile='on')
		else:
			_x =Variable(_ax, volatile=volatile)
		_t =Variable(_at, volatile=volatile)

		x = ChainerTensor(_x)
		t = ChainerTensor(_t)

		loss,accuracy = workout(x,t)


		Deel.trainCount+=1


		BatchTrainer.res_q.put((float(loss), float(accuracy)))
		del _x,_t
		del x,t
		del _ax,_at

def cnn_lstm_trainer(workout):
	Deel.trainCount=0
	while True:

		train_list = BatchTrainer.lstm_train		

		for i in range(len(train_list)):
			x = Deel.xp.asarray([read_image(train_list[i][0],False,True)])
			x = ChainerTensor(Variable(x,volatile='on'))
			x.use()

			result = workout(x,train_list[i][1])

			Deel.trainCount+=1



def isImageFile(path):
	root, ext = os.path.splitext(path)
	if ext=='.gif':
		return True
	if ext=='.png':
		return True
	if ext=='.jpeg':
		return True
	if ext=='.jpg':
		return True
	return False


def InputBatch(train='data/train.txt',val='data/test.txt'):
	Deel.train=train
	Deel.val = val

	root, ext = os.path.splitext(train)
	if ext==".txt":
		BatchTrainer.train_list = load(Deel.train,Deel.root)
		BatchTrainer.val_list = load(Deel.val,Deel.root)
		BatchTrainer.in_size=ImageNet.in_size
		BatchTrainer.mode='CNN'
	if ext==".tsv":
		lstm_pairs=[]
		for line in  open(train).readlines():
			data = line.rstrip().split('\t')
			lstm_pairs.append( data)
		BatchTrainer.lstm_train = lstm_pairs
		if isImageFile(lstm_pairs[0][0]):
			BatchTrainer.in_size=ImageNet.in_size
			BatchTrainer.mode='CNN-LSTM'
		else:
			BatchTrainer.mode='LSTM'

		vocab={}
		for i in range(len(lstm_pairs)):
			word = lstm_pairs[i][1]
			for char in word:
				if char not in vocab:
					vocab[char] = len(vocab)
		BatchTrainer.vocab = vocab




def BatchTrain(callback,callback2):
	global workout,checkout
	trainer = BatchTrainer()

	if BatchTrainer.mode=='CNN':
		feeder = threading.Thread(target=feed_data)
		feeder.daemon = True
		feeder.start()
		logger = threading.Thread(target=log_result)
		logger.daemon = True
		logger.start()	

		workout = callback
		checkout=callback2

		train_loop()
		feeder.join()
		logger.join()
	elif BatchTrainer.mode=='CNN-LSTM':
		cnn_lstm_trainer(callback)




def Show(x=None):
	if x==None:
		x = Tensor.context

	x.show()

def ShowLabels(x=None):
	if x==None:
		x = Tensor.context

	t = LabelTensor(x)

	t.show()
	return t

def GetLabels(x=None):
	if x==None:
		x = Tensor.context

	t = LabelTensor(x)

	return t.content


def Output(x=None,num_of_candidate=5):
	if x==None:
		x = Tensor.context

	out = x.output

	out.sort(cmp=lambda a, b: cmp(a[0], b[0]), reverse=True)
	for rank, (score, name) in enumerate(out[:num_of_candidate], start=1):
		print('#%d | %s | %4.1f%%' % (rank, name, score * 100))


