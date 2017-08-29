# import gym

# # wrappers allows gym to record performance
# from gym import wrappers

# # Sets which environment to get
# env = gym.make('Marvin-v0')

# # Sets wrapper location
# #env = wrappers.Monitor(env, '/tmp/marvin_exp1', force=True)

# # Prints out available actions
# print(env.action_space)

# print (env.observation_space)

# for i_episode in range(20):
# 	observation = env.reset()
# 	for i_episode in range(1000):

# 		env.render()

# 		# Starts the process
# 		print (observation)

# 		# Gets random action
# 		action = env.action_space.sample()

# 		# Starts action
# 		observation, reward, done, info = env.step(action)

# 		print ("Reward is %s" % (reward))

# 		# Renders to screen
# 		env.render()

# 		# Checks if still alive if not starts next episode
# 		if done:
# 			print("Episode finished")
# 			break


import gym

class Marvin:
	LEARNING_RATE = 0.01

	def __init__(self):
		self.env = gym.make('Marvin-v0')

	def play(self, episodes, render=False):
		for i_episode in range(episodes):
			observation = self.env.reset()
			while True:
				# Starts the process
				#print (observation)

				# Gets random action
				action = self.env.action_space.sample()

				# Starts action
				observation, reward, done, info = self.env.step(action)

				print ("Reward is %s" % (reward))

				# Renders to screen
				if render:
					self.env.render()

				# Checks if still alive if not starts next episode
				if done:
					print("Episode finished")
					break

marvin = Marvin()
marvin.play(10)


# import random
# import cPickle as pickle
# import numpy as np
# from evostra import EvolutionStrategy
# from model import Model
# import gym


# class Agent:

#     AGENT_HISTORY_LENGTH = 1
#     POPULATION_SIZE = 20
#     EPS_AVG = 1
#     SIGMA = 0.1
#     LEARNING_RATE = 0.01
#     INITIAL_EXPLORATION = 1.0
#     FINAL_EXPLORATION = 0.0
#     EXPLORATION_DEC_STEPS = 1000000

#     def __init__(self):
#         self.env = gym.make('BipedalWalker-v2')
#         self.model = Model()
#         self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE, self.SIGMA, self.LEARNING_RATE)
#         self.exploration = self.INITIAL_EXPLORATION


#     def get_predicted_action(self, sequence):
#         prediction = self.model.predict(np.array(sequence))
#         return prediction


#     def load(self, filename='weights.pkl'):
#         with open(filename,'rb') as fp:
#             self.model.set_weights(pickle.load(fp))
#         self.es.weights = self.model.get_weights()


#     def save(self, filename='weights.pkl'):
#         with open(filename, 'wb') as fp:
#             pickle.dump(self.es.get_weights(), fp)


#     def play(self, episodes, render=True):
#         self.model.set_weights(self.es.weights)
#         for episode in xrange(episodes):
#             total_reward = 0
#             observation = self.env.reset()
#             sequence = [observation]*self.AGENT_HISTORY_LENGTH
#             done = False
#             while not done:
#                 if render:
#                     self.env.render()
#                 action = self.get_predicted_action(sequence)
#                 observation, reward, done, _ = self.env.step(action)
#                 total_reward += reward
#                 sequence = sequence[1:]
#                 sequence.append(observation)
#             print "total reward:", total_reward


#     def train(self, iterations):
#         self.es.run(iterations, print_step=1)


#     def get_reward(self, weights):
#         total_reward = 0.0
#         self.model.set_weights(weights)

#         for episode in xrange(self.EPS_AVG):
#             observation = self.env.reset()
#             sequence = [observation]*self.AGENT_HISTORY_LENGTH
#             done = False
#             while not done:
#                 self.exploration = max(self.FINAL_EXPLORATION, self.exploration - self.INITIAL_EXPLORATION/self.EXPLORATION_DEC_STEPS)
#                 if random.random() < self.exploration:
#                     action = self.env.action_space.sample()
#                 else:
#                     action = self.get_predicted_action(sequence)
#                 observation, reward, done, _ = self.env.step(action)
#                 total_reward += reward
#                 sequence = sequence[1:]
#                 sequence.append(observation)

#         return total_reward/self.EPS_AVG

# agent = Agent()
# agent.play(10000, render=False)