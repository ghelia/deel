import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer.links import caffe
from chainer import computational_graph as c
from deel.tensor import *
from deel.network import *
import copy

from deel.deel import *
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
import hashlib
import datetime
import sys
import random
	 
from deel.agentServer import AgentServer
import deel.model.q_net
class DQN(Network):
	#lastAction = Action()
	policyFrozen = False
	epsilonDelta = 1.0 / 10 ** 4
	min_eps = 0.1

	actions = None
	reward=None

	def __init__(self,given_dim=(256,6,6),actions=[0,1,2],depth_image_dim=0):		
		super(DQN,self).__init__('Deep Q-learning Network')
		self.actions = actions
		self.age = 0

		self.depth_image_dim = depth_image_dim

		self.image_feature_dim = getDim(given_dim)+depth_image_dim

		print("shape:",self.image_feature_dim)

		self.time = 0
		self.epsilon = 1.0  # Initial exploratoin rate
		self.func = deel.model.q_net.QNet(Deel.gpu,self.actions,self.image_feature_dim)

	def actionAndLearn(self,x=None):
		if x is None:
			x=Tensor.context


		if AgentServer.mode == 'start':
			return self.start(x)
		elif AgentServer.mode == 'step':
			return self.step(x)
		elif AgentServer.mode == 'end':
			return self.end(x)

	 
	def start(self,x=None):
		if x is None:
			x=Tensor.context


		obs_array = x.content.data
		#print "sum",obs_array.sum()

		# Initialize State
		self.state = np.zeros((self.func.hist_size, self.image_feature_dim), dtype=np.uint8)
		self.state[0] = obs_array
		state_ = np.asanyarray(self.state.reshape(1, self.func.hist_size, self.image_feature_dim), dtype=np.float32)
		if Deel.gpu >= 0:
			state_ = cuda.to_gpu(state_)

		# Generate an Action e-greedy
		action, Q_now = self.func.e_greedy(state_, self.epsilon)
		returnAction = action

		# Update for next step
		self.lastAction = copy.deepcopy(returnAction)
		self.last_state = self.state.copy()
		self.last_observation = obs_array

		return returnAction

	def step(self,x=None):
		if x is None:
			x=Tensor.context

		obs_array = x.content.data
		#print "sum",obs_array.sum()
		obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

		# Compose State : 4-step sequential observation
		if self.func.hist_size == 4:
			self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_processed], dtype=np.uint8)
		elif self.func.hist_size == 2:
			self.state = np.asanyarray([self.state[1], obs_processed], dtype=np.uint8)
		elif self.func.hist_size == 1:
			self.state = np.asanyarray([obs_processed], dtype=np.uint8)
		else:
			print("self.DQN.hist_size err")

		state_ = np.asanyarray(self.state.reshape(1, self.func.hist_size, self.image_feature_dim), dtype=np.float32)
		if Deel.gpu >= 0:
			state_ = cuda.to_gpu(state_)

		# Exploration decays along the time sequence
		if self.policyFrozen is False:  # Learning ON/OFF
			if self.func.initial_exploration < self.time:
				self.epsilon -= self.epsilonDelta
				if self.epsilon < self.min_eps:
					self.epsilon = self.min_eps
				eps = self.epsilon
			else:  # Initial Exploation Phase
				print("Initial Exploration : %d/%d steps" % (self.time, self.func.initial_exploration))
				eps = 1.0
		else:  # Evaluation
			print("Policy is Frozen")
			eps = 0.05

		# Generate an Action by e-greedy action selection
		action, Q_now = self.func.e_greedy(state_, eps)

		return self,action, eps, Q_now, obs_array
	def step_after(self,reward, action, eps, q_now, obs_array):

		'''
		Step after
		'''

		self.reward = reward
		# Learning Phase
		if self.policyFrozen is False:  # Learning ON/OFF
			self.func.stock_experience(self.time, self.last_state, self.lastAction, reward, self.state, False)
			self.func.experience_replay(self.time)

		# Target model update
		if self.func.initial_exploration < self.time and np.mod(self.time, self.func.target_model_update_freq) == 0:
			print('Model Updated')
			self.func.target_model_update()

		# Simple text based visualization
		if Deel.gpu >= 0:
			q_max = np.max(q_now.get())
		else:
			q_max = np.max(q_now)
		print('Step %d/ACT %d/R %.1f/EPS %.6f/Q_max %3f' % (
			self.time, self.func.action_to_index(action), reward, eps, q_max))

		# Updates for next step
		self.last_observation = obs_array

		if self.policyFrozen is False:
			self.lastAction = copy.deepcopy(action)
			self.last_state = self.state.copy()
			self.time += 1

			#if self.time % 1000 == 0:
			#	pickle.dump(self.func.model, open("cash/qnet_%05d.pkl"%self.time, 'wb'))
		

	def end(self,x):
		reward = self.reward
		print('episode finished: REWARD %.1f / EPSILON %.5f' % (reward, self.epsilon))

		# Learning Phase
		if self.policyFrozen is False:  # Learning ON/OFF
			self.func.stock_experience(self.time, self.last_state, self.lastAction, reward, self.last_state,
										True)
			self.func.experience_replay(self.time)

		# Target model update
		if self.func.initial_exploration < self.time and np.mod(self.time, self.func.target_model_update_freq) == 0:
			print('Model Updated')
			self.func.target_model_update()

		# Time count
		if self.policyFrozen is False:
			self.time += 1

		self.age += 1
		if self.age % 10 ==0:
			self.save('dqn%04d.pkl'%self.age)			

	def save(self,filename):
		pickle.dump(self.func.model, open(filename, 'wb')) 

	def load(self,filename):
		self.func.model = pickle.load(open(filename,'rb'))

		

