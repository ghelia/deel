import cPickle as pickle
import copy
import os
import numpy as np
from chainer import cuda
from rlglue.types import Action

from network import *
from model.q_net import QNet


class Agent(object):
    lastAction = Action()
    policyFrozen = False
    dim = 256 * 6 * 6
    epsilonDelta = 1.0 / 10 ** 4
    min_eps = 0.05
    model_loaded = False
    cnn_feature_extractor = 'cash/alexnet.pickle'

    def agent_init(self, use_gpu):
        self.use_gpu = Deel.gpu
        '''
        if os.path.exists(self.cnn_feature_extractor):
            print("loading... " + self.cnn_feature_extractor),
            self.feature_extractor = pickle.load(open(self.cnn_feature_extractor))
            print("done")
        else:
            self.feature_extractor = AlexnetFeatureExtractor(self.use_gpu)
            pickle.dump(self.feature_extractor, open(self.cnn_feature_extractor, 'w'))
            print("pickle.dump finished")
        '''

        self.feature_extractor = AlexNet()

        # Some initializations for rlglue
        self.lastAction = Action()

        self.time = 0
        self.epsilon = 1.0  # Initial exploratoin rate

        # Pick a DQN from DQN_class
        #self.q_net = QNet(self.use_gpu)

    def agent_start(self, observation):
        # # Preprocess
        obs_array = self.feature_extractor.feature(observation)

        # Initialize State
        self.state = np.zeros((self.q_net.hist_size, self.dim), dtype=np.uint8)
        self.state[0] = obs_array
        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Generate an Action e-greedy
        returnAction = Action()
        action, Q_now = self.q_net.e_greedy(state_, self.epsilon)
        returnAction.intArray = [action]

        # Update for next step
        self.lastAction = copy.deepcopy(returnAction)
        self.last_state = self.state.copy()
        self.last_observation = obs_array

        return returnAction

    def agent_step(self, reward, observation):
        # # Preproces
        obs_array = self.feature_extractor.feature(observation)
        obs_processed = np.maximum(obs_array, self.last_observation)  # Take maximum from two frames

        # Compose State : 4-step sequential observation
        if self.q_net.hist_size == 4:
            self.state = np.asanyarray([self.state[1], self.state[2], self.state[3], obs_processed], dtype=np.uint8)
        elif self.q_net.hist_size == 1:
            self.state = np.asanyarray([obs_processed], dtype=np.uint8)
        else:
            print("self.DQN.hist_size err")

        state_ = np.asanyarray(self.state.reshape(1, self.q_net.hist_size, self.dim), dtype=np.float32)
        if self.use_gpu >= 0:
            state_ = cuda.to_gpu(state_)

        # Exploration decays along the time sequence
        if self.policyFrozen is False:  # Learning ON/OFF
            if self.q_net.initial_exploration < self.time:
                self.epsilon -= self.epsilonDelta
                if self.epsilon < self.min_eps:
                    self.epsilon = self.min_eps
                eps = self.epsilon
            else:  # Initial Exploation Phase
                print "Initial Exploration : %d/%d steps" % (self.time, self.q_net.initial_exploration)
                eps = 1.0
        else:  # Evaluation
            print "Policy is Frozen"
            eps = 0.05

        # Generate an Action by e-greedy action selection
        returnAction = Action()
        action, Q_now = self.q_net.e_greedy(state_, eps)
        returnAction.intArray = [action]

        return returnAction, action, eps, Q_now, obs_array, returnAction

    def agent_step_after(self, reward, observation, action, eps, Q_now, obs_array, returnAction):
        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.state, False)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.q_net.target_model_update()

        # Simple text based visualization
        if self.use_gpu >= 0:
            print 'Step %d/ACT %d/R %.1f/EPS %.6f/Q_max %3f' % (
                self.time, self.q_net.action_to_index(action), reward, eps, np.max(Q_now.get()))
        else:
            print 'Step %d/ACT %d/R %.1f/EPS %.6f/Q_max %3f' % (
                self.time, self.q_net.action_to_index(action), reward, eps, np.max(Q_now))

        # Updates for next step
        self.last_observation = obs_array

        if self.policyFrozen is False:
            self.lastAction = copy.deepcopy(returnAction)
            self.last_state = self.state.copy()
            self.time += 1

    def agent_end(self, reward):  # Episode Terminated
        print 'episode finished: REWARD %.1f / EPSILON %.5f' % (reward, self.epsilon)

        # Learning Phase
        if self.policyFrozen is False:  # Learning ON/OFF
            self.q_net.stock_experience(self.time, self.last_state, self.lastAction.intArray[0], reward, self.last_state,
                                        True)
            self.q_net.experience_replay(self.time)

        # Target model update
        if self.q_net.initial_exploration < self.time and np.mod(self.time, self.q_net.target_model_update_freq) == 0:
            print "########### MODEL UPDATED ######################"
            self.q_net.target_model_update()

        # Time count
        if self.policyFrozen is False:
            self.time += 1
