import random

from apps.cart.debug_environment import CartDebugEnvironment
from apps.cart.environment import CartEnvironment
from apps.utils.gym_utils import Epsilon


class CartAiDebugEnvironment(CartDebugEnvironment):

	def __init__(self, env: CartEnvironment, model, epsilon: Epsilon = None):
		super().__init__(env)
		self.model = model
		self.epsilon = epsilon or Epsilon.none()

	def reset_environment(self):
		super().reset_environment()
		self.epsilon.decay()

	def read_user_input(self):
		if self.epsilon.decide_greedy():
			return self.env.random_input()
		return self.env.translate_prediction_to_input(self.model(self.env.observe()))
