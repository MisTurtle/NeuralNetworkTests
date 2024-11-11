from abc import ABC, abstractmethod

import keras
import numpy as np
import pymunk
import pymunk.pygame_util


class BaseEnvironment(ABC):
	"""
	An environment contains the landscape for a simulation and allows to setup objects like the agent, obstacles and action space
	From that environment, two types of programs can be built :
		- A Debug environment, which allows a human user to try out the space where the AI will evolve in
		- A Gym, which provides methods to standardize the environment observations
	"""

	STATE_RUNNING = 0
	STATE_DIED = 1

	def __init__(self):
		self._state = BaseEnvironment.STATE_RUNNING

	def get_state(self):
		return self._state

	def set_state(self, state: int):
		self._state = state

	def new_episode_case(self):
		"""
		Reset the environment, probably because of the agent's death or a time limit reached
		"""
		self.set_state(self.STATE_RUNNING)
		self.reset_environment()

	def play_step(self, actions, dt: float):
		"""
		Play a physics step with actions as input (Actions can be passed from the user or a neural network)
		"""
		self.process_input(actions, dt)

	@abstractmethod
	def get_environment_name(self) -> str:
		"""
		Give a cool name to the environment
		"""
		pass

	@abstractmethod
	def process_input(self, actions, dt):
		"""
		Interacts with the environment according to the provided input
		"""
		pass

	@abstractmethod
	def create_model(self) -> keras.Model:
		"""
		Creates a new model to be used to solve the environment
		"""
		pass

	@abstractmethod
	def translate_prediction_to_input(self, prediction):
		"""
		Translates a neural network prediction into inputs as a human user would perform
		"""
		pass

	@abstractmethod
	def get_action_space_size(self) -> int:
		"""
		How many actions can be performed by the network
		"""
		pass

	@abstractmethod
	def get_input_space_size(self) -> int:
		"""
		How many inputs are fed into the network
		"""
		pass

	@abstractmethod
	def observe(self) -> np.array:
		"""
		Compiles relevant state information about the simulation (probably in order to be fed as input to the neural network)
		"""
		pass

	@abstractmethod
	def random_input(self):
		"""
		Get a random set of inputs matching the expected NN size
		"""
		pass

	@abstractmethod
	def get_environment_size(self) -> tuple[int, int]:
		"""
		Environment size in which the simulation occurs (also used as pygame window size)
		"""
		pass

	@abstractmethod
	def setup_environment(self):
		"""
		Sets up the environment for the first time
		"""
		pass

	@abstractmethod
	def reset_environment(self):
		"""
		Resets the environment, probably to prepare for a new episode case
		"""

	@abstractmethod
	def compute_reward(self) -> float:
		"""
		Computes a reward based on the state of the game
		"""
		pass

	@abstractmethod
	def draw(self, screen):
		"""
		Draws the environment's state onto a pygame canvas
		"""
		pass


class PhysicsEnvironment(BaseEnvironment, ABC):
	"""
	Environment relying on pymunk physics
	"""

	def __init__(self):
		super().__init__()
		self._space = pymunk.Space()
		self._space.gravity = (0, -981)
		self._draw_options = None
		pymunk.pygame_util.positive_y_is_up = True

	def get_space(self) -> pymunk.Space:
		return self._space

	def play_step(self, actions, dt: float):
		super().play_step(actions, dt)
		self.get_space().step(dt)

	def draw(self, screen):
		if self._draw_options is None:
			self._draw_options = pymunk.pygame_util.DrawOptions(screen)
		self.get_space().debug_draw(self._draw_options)
