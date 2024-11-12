import random
from abc import ABC

import numpy as np
import pymunk

from apps.utils.environment import PhysicsEnvironment


class CartEnvironment(PhysicsEnvironment, ABC):
	"""
	This environment contains a cart stuck on a horizontal line, as well as a target on that same line
	I know, it's a very, very simple environment, but I just can't figure out for the life of me why more complete ones are not yielding any results !!! (Yeah, I'm desperate)
	Anyways, if this one doesn't work either, I'm gonna cry, literally

	Yeepee, it finally worked, just needed some tweaking with penalty for smashing the screen borders
	"""

	def __init__(self):
		super().__init__()
		es = self.get_environment_size()

		self.cart_size, self.cart_weight = (40, 25), 1

		self.guide = pymunk.Segment(pymunk.Body(body_type=pymunk.Body.STATIC), (0, es[1] / 2), (es[0], es[1] / 2), radius=1)
		self.guide.sensor = True

		self.target = pymunk.Vec2d(0, 0)
		self.target_radius = 10

		self.cart_body = pymunk.Body(self.cart_weight, pymunk.moment_for_box(self.cart_weight, self.cart_size))
		self.cart_poly = pymunk.Poly.create_box(self.cart_body, self.cart_size)

		self.get_space().iterations = 20

	def translate_prediction_to_input(self, prediction):
		return np.argmax(prediction)

	def get_action_space_size(self) -> int:
		return 2  # Left or right

	def random_input(self) -> int:
		return int(random.random() < 0.5)

	def get_environment_size(self) -> tuple[int, int]:
		return 1500, 450

	def get_environment_name(self) -> str:
		return "Cart"

	def process_input(self, actions, dt):
		force_mag = 10
		if actions == 0:
			if self.cart_body.velocity.x > 0:
				force_mag *= 2
			self.cart_body.apply_impulse_at_local_point((-force_mag, 0))
		elif actions == 1:
			if self.cart_body.velocity.x < 0:
				force_mag *= 2
			self.cart_body.apply_impulse_at_local_point((force_mag, 0))

		if self.target_reached() or self.cart_body.position.x < self.cart_size[0] + 5 or self.cart_body.position.x > self.get_environment_size()[0] - self.cart_size[0] - 5:
			self.set_state(self.STATE_DIED)

	def target_reached(self) -> bool:
		dist = abs(self.cart_body.position.x - self.target.x)
		return dist < self.target_radius + self.cart_size[0] / 2

	def create_model(self):
		from keras import Sequential
		from keras.src.layers import Dense
		from keras.src.optimizers import Adam
		model = Sequential([
			Dense(64, activation='relu', kernel_initializer='random_normal', input_shape=(self.get_input_space_size(),)),
			Dense(32, activation='relu', kernel_initializer='random_normal'),
			Dense(self.get_action_space_size(), activation='softmax', kernel_initializer='random_normal')
		])
		model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
		return model

	def reset_environment(self):
		self.cart_body.velocity = 0, 0
		once = False
		while not once or self.target_reached():
			self.cart_body.position = random.random() * self.get_environment_size()[0], self.get_environment_size()[1] / 2
			self.target = pymunk.Vec2d(random.random() * self.get_environment_size()[0], self.get_environment_size()[1] / 2)
			once = True

	def setup_environment(self):
		width, height = self.get_environment_size()

		groove_joint_1 = pymunk.GrooveJoint(self.guide.body, self.cart_body, (0, height/2), (width, height/2), (-self.cart_size[0]/2, 0))
		groove_joint_1.error_bias = 0.00001
		groove_joint_2 = pymunk.GrooveJoint(self.guide.body, self.cart_body, (0, height/2), (width, height/2), (self.cart_size[0]/2, 0))
		groove_joint_2.error_bias = 0.00001

		self.get_space().add(self.guide, self.guide.body, self.cart_body, self.cart_poly)
		self.get_space().add(groove_joint_1, groove_joint_2)


class CartEnvironment_V1(CartEnvironment):

	def get_input_space_size(self) -> int:
		return 3

	def observe(self) -> np.array:
		return np.array([[
			(self.cart_body.position.x / self.get_environment_size()[0]) - 0.5,
			(self.target.x / self.get_environment_size()[0]) - 0.5,
			self.cart_body.velocity.x
		]])

	def compute_reward(self) -> float:
		if self.target_reached():
			return 10
		distance_penalty = -abs(self.cart_body.position.x - self.target.x) / self.get_environment_size()[0]
		speed_penalty = self.cart_body.velocity.x / 500 * (-1 if self.cart_body.position.x > self.target.x else 1)
		return distance_penalty + speed_penalty


class CartEnvironment_V2(CartEnvironment):

	def get_input_space_size(self) -> int:
		return 2

	def observe(self) -> np.array:
		return np.array([[
			(self.cart_body.position.x / self.get_environment_size()[0]) - 0.5,
			(self.target.x / self.get_environment_size()[0]) - 0.5
		]])

	def compute_reward(self) -> float:
		if self.target_reached():
			return 10
		elif self.get_state() != self.STATE_RUNNING:
			return -10
		distance_penalty = -abs(self.cart_body.position.x - self.target.x) / self.get_environment_size()[0]
		speed_penalty = self.cart_body.velocity.x / 500 * (-1 if self.cart_body.position.x > self.target.x else 1)
		return distance_penalty + speed_penalty
