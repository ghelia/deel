import chainer.functions as F
from chainer import Variable, FunctionSet, optimizers
from chainer.links import caffe
from chainer import computational_graph
from chainer import cuda
from chainer import optimizers
from chainer import serializers
from tensor import *
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
import datetime
import sys
import random

xp = np


class Deel(object):
	singlton = None
	train = None
	val = None
	root = '.'
	epoch=100
	gpu=-1
	mean=None
	labels=None
	optimizer_lr=0.1
	def __init__(self,gpu=-1):
		global xp
		self.singleton = self
		self.gpu=gpu
		if gpu>=0:
			cuda.get_device(gpu).use()
			xp = cuda.cupy if gpu >= 0 else np

	@staticmethod
	def getInstance():
		return Deel.singleton

