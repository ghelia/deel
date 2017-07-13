import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers
from chainer.links import caffe
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from .tensor import *
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


class Deel(object):
	singlton = None
	train = False
	val = None
	root = '.'
	epoch=1000
	gpu=-1
	mean=None
	labels=None
	xp = np
	optimizer_lr=0.1
	trainCount=0
	lstm_train=None
	defferedTuning=False
	def __init__(self,gpu=-1):
		Deel.singleton = self
		Deel.gpu=gpu
		if gpu>=0:
			cuda.get_device(gpu).use()
			Deel.xp = cuda.cupy if gpu >= 0 else np

	@staticmethod
	def getInstance():
		return Deel.singleton

'''
	Trainer
'''
class BatchTrainer(object):
	batchsize=32
	val_batchsize=32
	data_q=None
	res_q=None
	loaderjob=20
	train_list=None
	val_list=None
	lstm_train=None
	num_of_vocab=0
	mode='CNN'
	vocab=None
	in_size=256
	def __init__(self,in_size=256):
		BatchTrainer.data_q = queue.Queue(maxsize=1)
		BatchTrainer.res_q = queue.Queue()

