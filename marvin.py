##############################

	#My Current Try

##############################

import gym
import time
import numpy as np

class Marvin:
	LEARNING_RATE = 0.01

	def __init__(self):
		self.env = gym.make('Marvin-v0')
		self.model = Model()

	def get_action(self, state):
			hl = np.matmul(state, self.model.matrix['W1'])
			hl = np.tanh(hl)
			action = np.matmul(hl, self.model.matrix['W2'])
			action = np.tanh(action)

			return action

	def start(self, episodes, render=False):
		for i in range(episodes):
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
					break
		return total_reward


	def play(self, episodes, render=False):
		for i_episode in range(episodes):
			observation = self.env.reset()
			while True:
				# Starts the process
				#print (observation)

				# Gets random action
				action = self.env.action_space.sample()

				print (self.env.observation_space[0])

				# Starts action
				observation, reward, done, info = self.env.step(self.env.observation_space.low)

				print ("Reward is %s" % (reward))

				# Renders to screen
				if render:
					self.env.render()

				# Checks if still alive if not starts next episode
				if done:
					print("Episode finished")
					break

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
marvin.start(1000, render=True)
#marvin.play(10, render=True)