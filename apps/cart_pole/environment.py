import math
import random
from abc import ABC

import keras
import numpy as np
import pymunk
from keras import Sequential
from keras.src.layers import Dense
from keras.src.optimizers import Adam

from apps.utils.environment import PhysicsEnvironment


class CartPoleEnvironment(PhysicsEnvironment, ABC):

	def __init__(self):
		super().__init__()
		es = self.get_environment_size()

		self.cart_size, self.cart_weight = (40, 25), 1
		self.pole_size, self.pole_weight = (5, 70), 0.1
		self.max_angle = math.pi * 20 / 180

		self.guide = pymunk.Segment(pymunk.Body(body_type=pymunk.Body.STATIC), (0, es[1] / 2), (es[0], es[1] / 2), radius=1)
		self.guide.sensor = True

		self.cart_body = pymunk.Body(self.cart_weight, pymunk.moment_for_box(self.cart_weight, self.cart_size))
		self.cart_poly = pymunk.Poly.create_box(self.cart_body, self.cart_size)

		self.pole_body = pymunk.Body(self.pole_weight, pymunk.moment_for_box(self.pole_weight, self.pole_size))
		self.pole_poly = pymunk.Poly.create_box(self.pole_body, self.pole_size)
		self.pole_poly.sensor = True

		self.get_space().iterations = 20

	def translate_prediction_to_input(self, prediction):
		return np.argmax(prediction)

	def get_action_space_size(self) -> int:
		return 2

	def get_input_space_size(self) -> int:
		return 4

	def observe(self) -> np.array:
		return np.array([[
			(self.cart_body.position.x / self.get_environment_size()[0]) - 0.5,
			self.cart_body.velocity.x / 200,
			self.pole_body.angle,
			self.pole_body.angular_velocity
		]])

	def random_input(self) -> int:
		return int(random.random() < 0.5)

	def get_environment_size(self) -> tuple[int, int]:
		return 1500, 450

	def setup_environment(self):
		width, height = self.get_environment_size()
		pivot = pymunk.PivotJoint(self.cart_body, self.pole_body, (0, self.cart_size[1] / 2), (0, -self.pole_size[1] / 2))

		groove_joint_1 = pymunk.GrooveJoint(self.guide.body, self.cart_body, (0, height/2), (width, height/2), (-self.cart_size[0]/2, 0))
		groove_joint_1.error_bias = 0.00001
		groove_joint_2 = pymunk.GrooveJoint(self.guide.body, self.cart_body, (0, height/2), (width, height/2), (self.cart_size[0]/2, 0))
		groove_joint_2.error_bias = 0.00001

		self.get_space().add(self.guide, self.guide.body, self.cart_body, self.cart_poly, self.pole_body, self.pole_poly)
		self.get_space().add(pivot, groove_joint_1, groove_joint_2)

	def reset_environment(self):
		def get_random_value() -> float:
			return random.random() / 9

		self.cart_body.velocity = get_random_value(), 0
		self.cart_body.position = self.get_environment_size()[0] / 2 + get_random_value(), self.get_environment_size()[1] / 2
		self.pole_body.position = self.cart_body.position.x, self.cart_body.position.y + self.pole_size[1] / 2 + self.cart_size[1] / 2
		self.pole_body.angle = get_random_value()
		self.pole_body.angular_velocity = 0
		self.pole_body.velocity = 0, 0

	def process_input(self, actions, dt):
		force_mag = 200
		if actions == 0:
			self.cart_body.apply_force_at_local_point((-force_mag * self.cart_weight, 0))
		elif actions == 1:
			self.cart_body.apply_force_at_local_point((force_mag * self.cart_weight, 0))
		if not -self.max_angle < self.pole_body.angle < self.max_angle:
			self.set_state(self.STATE_DIED)

	def create_model(self) -> keras.Model:
		model = Sequential([
			Dense(64, activation='relu', kernel_initializer='random_normal', input_shape=(self.get_input_space_size(),)),
			Dense(64, activation='relu', kernel_initializer='random_normal'),
			Dense(self.get_action_space_size(), activation='linear', kernel_initializer='random_normal')
		])
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
		return model


class CartPoleEnvironment_V1(CartPoleEnvironment):

	def get_environment_name(self) -> str:
		return "CartPole_V1"

	def compute_reward(self) -> float:
		# No distance reward
		if self.get_state() == self.STATE_DIED:
			return -5
		angle_reward = 1 - abs(self.pole_body.angle) / self.max_angle
		return angle_reward


class CartPoleEnvironment_V2(CartPoleEnvironment):

	def get_environment_name(self) -> str:
		return "CartPole_V2"

	def compute_reward(self) -> float:
		# This somehow didn't fix the problem where the cart would drift away gradually
		if self.get_state() == self.STATE_DIED:
			return -5
		angle_reward = 1 - abs(self.pole_body.angle) / self.max_angle  # [0; 1]
		distance_reward = 1 - 10 * abs((self.cart_body.position.x / self.get_environment_size()[0]) - 0.5)  # [0; 1]
		return (angle_reward + distance_reward) / 2


class CartPoleEnvironment_V3(CartPoleEnvironment):

	def __init__(self):
		super().__init__()
		self.max_distance = self.get_environment_size()[0] / 6
		self.min_x, self.max_x = self.get_environment_size()[0] / 2 - self.max_distance, self.get_environment_size()[0] / 2 + self.max_distance

	def get_environment_name(self) -> str:
		return "CartPole_V3"

	def process_input(self, actions, dt):
		super().process_input(actions, dt)
		if self.get_state() != self.STATE_DIED:
			if not self.min_x < self.cart_body.position.x < self.max_x:
				self.set_state(self.STATE_DIED)

	def compute_reward(self) -> float:
		# This somehow didn't fix the problem where the cart would drift away gradually
		if self.get_state() == self.STATE_DIED:
			return -10
		angle_reward = 1 - abs(self.pole_body.angle) / self.max_angle  # [0; 1]
		distance_reward = 1 - 2 * abs((self.cart_body.position.x / self.get_environment_size()[0]) - 0.5)  # [0; 1]
		return (angle_reward + distance_reward) / 2



