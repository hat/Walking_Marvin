##############################

	#My Current Try

##############################

# PROBLEMS: Might have to many nested objects

import gym
import time
import numpy as np

class Marvin:
	LEARNING_RATE = 0.01
	npop = 40
	sigma = 0.3

	def __init__(self):
		self.env = gym.make('Marvin-v0')
		self.model = Model()

	def get_action(self, state):
			hl = np.matmul(state, self.model.matrix['W1'])
			hl = np.tanh(hl)
			action = np.matmul(hl, self.model.matrix['W2'])
			action = np.tanh(action)

			return action

	def start(self, render=False):
		state = self.env.reset()
		total_reward = 0
		for j in range(10000):
			if render:
				self.env.render()
			action = self.get_action(state)
			state, reward, done, info = self.env.step(action)
			total_reward += reward

			print ("State: %s Reward: %s Done %s Info %s" % (state, reward, done, info))

			if done:
				print ("Reward: %s" % (reward))
				break
		return total_reward

	def train(self, sims=10000):
		for i in range(sims):
			model_n = {}
			for r, c in self.model.matrix.iteritems():
				model_n[r] = np.random.randn(Marvin.npop, c.shape[0], c.shape[1])
			R = np.zeros(Marvin.npop)

			for j in range(Marvin.npop):
				test_model = {}
				for r, c in self.model.matrix.iteritems():
					print ("R is: %s" % (r))
					test_model[r] = c + Marvin.sigma * model_n[r][j]
				R[j] = Marvin.start(self)

			A = (R - np.mean(R)) / np.std(R)

			for i in self.model.matrix:
				self.model.matrix[i] = self.model.matrix[i] + alpha / (Marvin.npop * Marvin.sigma) * np.dot(model_n[i].transpose(1, 2, 0), A)

			cur_reward = start(self)
			aver_reward = aver_reward * 0.9 + cur_reward * 0.1 if aver_reward != None else cur_reward

class Model:

	def __init__(self):
		hl_size = 100
		t = int( time.time() * 1000.0 )
		np.random.seed( ((t & 0xff000000) >> 24) +
             			((t & 0x00ff0000) >>  8) +
             			((t & 0x0000ff00) <<  8) +
             			((t & 0x000000ff) << 24)   )
		self.matrix = {}
		self.matrix['W1'] = np.random.randn(24, hl_size) / np.sqrt(24)
		self.matrix['W2'] = np.random.randn(hl_size, 4) / np.sqrt(hl_size)

marvin = Marvin()
#marvin.start(render=True)
marvin.train()
#marvin.play(10, render=True)