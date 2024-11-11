import random

from apps.cart_pole.debug_environment import CartPoleDebugEnvironment
from apps.cart_pole.environment import CartPoleEnvironment
from apps.utils.gym_utils import Epsilon


class CartPoleAiDebugEnvironment(CartPoleDebugEnvironment):

	def __init__(self, env: CartPoleEnvironment, model, epsilon: Epsilon = None):
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
