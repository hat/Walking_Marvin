#!/usr/bin/env python2.7

# Project Metadata
__author__ = "Tony Hendrick, and Giacomo Guiulfo"
__copyright__ = "Copyright 2017, 42 Sillcon Valley"
__credits__ = ["Sasha Fedorova", "Gaetan Juvin"]
__license__ = "MIT"
__version__ = "0.4.2"
__maintainer__ = "The Vogons"
__email__ = "marvin@42.us.org"
__status__ = "Production"

# Imports
import gym
import time
import numpy as np
#import cPickle as pickle
import pickle
from flags import MarvinFlags

# Start of class of defintion
class Marvin:
	LOAD_FILE = False
	#LEARNING_R
	ATE = 0.01
	version = 1
	npop = 40
	sigma = 0.3
	alpha = 0.06
	aver_reward = None
	best_reward = -500.0

	def __init__(self, flags):
		self.env = gym.make('Marvin-v0')
		if flags.load:
			self.model = pickle.load(open(flags.load, 'rb'))
		else:
			hl_size = 100
			t = int( time.time() * 1000.0 )
			np.random.seed( ((t & 0xff000000) >> 24) +
	             			((t & 0x00ff0000) >>  8) +
	             			((t & 0x0000ff00) <<  8) +
	             			((t & 0x000000ff) << 24)   )
			self.model = {}
			self.model['W1'] = np.random.randn(24, hl_size) / np.sqrt(24)
			self.model['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size)
			# print(self.model['W1'])
			# print(self.model['W2'])

	def get_action(self, state, model):
			hl = np.matmul(state, model['W1'])
			hl = np.tanh(hl)
			action = np.matmul(hl, model['W2'])
			action = np.tanh(action)

			return action

	def start(self, flags, render=False):
		average_reward = 0
		for i in range(flags.plays):
			state = self.env.reset()
			total_reward = 0
			for j in range(flags.actions):
				if render:
					self.env.render()
				action = self.get_action(state, self.model)
				state, reward, done, info = self.env.step(action)
				total_reward += reward
				if done:
					break
			average_reward = (average_reward + total_reward) / (i + 1)
			if flags.info:
				print ("Total Reward: %s, Average Reward: %s" % (total_reward, average_reward))

	def test_train(self, model, render=False):
		state = self.env.reset()
		total_reward = 0
		for j in range(10000):
			if render:
				self.env.render()
			action = self.get_action(state, model)
			state, reward, done, info = self.env.step(action)
			total_reward += reward

			#print ("State: %s Reward: %s Done %s Info %s" % (state, reward, done, info))

			if done:
				#print ("Reward: %s" % (reward))
				break
		return total_reward

	def train(self, flags, sims=10000):
		while True:
			model_n = {}
			for r, c in self.model.iteritems():
				model_n[r] = np.random.randn(Marvin.npop, c.shape[0], c.shape[1])
			R = np.zeros(Marvin.npop)

			for j in range(Marvin.npop):
				test_model = {}
				for r, c in self.model.iteritems():
					#print ("R is: %s" % (r))
					test_model[r] = c + Marvin.sigma * model_n[r][j]
				R[j] = Marvin.test_train(self, test_model)

			A = (R - np.mean(R)) / np.std(R)

			for i in self.model:
				self.model[i] = self.model[i] + Marvin.alpha / (Marvin.npop * Marvin.sigma) * np.dot(model_n[i].transpose(1, 2, 0), A)

			cur_reward = Marvin.test_train(self, self.model)
			self.aver_reward = self.aver_reward * 0.9 + cur_reward * 0.1 if self.aver_reward != None else cur_reward
			print ("Current reward: %s Average reward: %s" % (cur_reward, self.aver_reward))
			if flags.save and cur_reward > self.best_reward:
				best_reward = cur_reward
				pickle.dump(self.model, open(flags.save, 'wb'))
# End of class definiton

# Main entry point of the program
if __name__ == "__main__":
	flags = MarvinFlags()
	marvin = Marvin(flags)
	if flags.walk:
		marvin.start(flags, render=True)
	else:
		marvin.train(flags)
		marvin.start(flags, render=True)
# End of program
