from abc import ABC, abstractmethod

import keras
import numpy as np
import tensorflow as tf

from apps.utils import gym_utils
from apps.utils.environment import BaseEnvironment
from apps.utils.gym_utils import ReplayBuffer


class LearningMode(ABC):

	def __init__(self, env: BaseEnvironment):
		self.env = env

	@abstractmethod
	def get_model(self) -> keras.Model:
		pass

	def get_action(self, state):
		"""
		Predict an action to perform given a state of observation
		"""
		return self.env.translate_prediction_to_input(self.get_model()(state))

	@abstractmethod
	def step(self) -> bool:
		"""
		Plays a simulation step with additional actions such as feeding the replay memory
		:returns: Whether the environment terminated
		"""
		pass

	@abstractmethod
	def train(self):
		"""
		Training method
		"""
		pass

	def load(self, path: str):
		self.get_model().load_weights(path)

	def save(self, path: str):
		self.get_model().save(path)


class DQN(LearningMode):

	def __init__(self, env: BaseEnvironment, memory: ReplayBuffer, gamma: float = 0.99, update_period: int = 1000):
		super().__init__(env)
		self.memory = memory
		self.gamma = gamma
		self.update_period = update_period
		self.q_model = env.create_model()
		self.target_q_model = env.create_model()
		self.ticks = 0

	def get_model(self) -> keras.Model:
		return self.q_model

	def train(self):
		sample = self.memory.np_sample()
		if sample is None:
			return

		states, actions, rewards, next_states, ends = sample
		states = np.reshape(states, (self.memory.batch_size, self.env.get_input_space_size()))
		next_states = np.reshape(next_states, (self.memory.batch_size, self.env.get_input_space_size()))

		state_predictions = self.q_model(states)
		next_state_predictions = self.target_q_model(next_states)

		max_q_values = tf.reduce_max(next_state_predictions, axis=1)
		target_values = rewards + (1 - ends) * self.gamma * max_q_values

		target_tensor = state_predictions.numpy()
		target_tensor[np.arange(self.memory.batch_size), actions] = target_values

		self.q_model.fit(states, target_tensor, batch_size=self.memory.batch_size, epochs=1, verbose=0)

		self.ticks += 1
		if self.ticks % self.update_period == 0:
			self.target_q_model.set_weights(self.q_model.get_weights())

	def step(self) -> bool:
		# Observe current state
		state = self.env.observe()
		action = self.get_action(state)

		# Tick the clock
		self.env.play_step(action, gym_utils.TIME_STEP)

		# Observe next state
		next_state, reward, ends = self.env.observe(), self.env.compute_reward(), self.env.get_state() == BaseEnvironment.STATE_DIED

		# Remember
		self.memory.remember(state, action, reward, next_state, int(ends))

		return ends

	def load(self, path: str):
		super().load(path)
		self.target_q_model.set_weights(self.q_model.get_weights())


