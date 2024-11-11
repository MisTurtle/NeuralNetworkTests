import abc
import os.path
from datetime import datetime

import pygame
import pymunk.pygame_util

from apps.utils.environment import BaseEnvironment

pygame.init()
pygame.font.init()


class SimplePygameDebugEnvironment(abc.ABC):

	PLAY_SPEED = [1, 1.25, 1.5, 2, 0.25, 0.5, 0.75]

	def __init__(self, env: BaseEnvironment):
		self.env = env
		self.screen = pygame.display.set_mode(env.get_environment_size())
		self.clock = pygame.time.Clock()
		self.font = pygame.font.SysFont('Tahoma', 11)
		self.running = True
		self.dt = 0.02

		self.step_by_step = False
		self.play_next_step = False  # Debug control : Key 6
		self.playing_speed = 0  # Debug control : Keys 9 and 2

		self.fill_color = 255, 255, 255
		self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
		env.setup_environment()

	def get_env(self):
		return self.env

	def handle_user_action(self, evt: pygame.event.Event):
		"""
		Used to perform actions when a key happens to be pressed on this very frame
		"""
		if evt.key == pygame.K_RETURN:
			self.env.new_episode_case()
		elif evt.key == pygame.K_q:
			self.running = False
		elif evt.key == pygame.K_p:
			if not os.path.exists("../../captures"):
				os.mkdir("../../captures")
			pygame.image.save(self.screen, "../../captures/" + datetime.today().strftime('%Y%m%d_%H%M%S.png'))
		elif evt.key == pygame.K_KP5:
			self.step_by_step = not self.step_by_step
		elif self.step_by_step and evt.key == pygame.K_KP6:
			self.play_next_step = True
		elif evt.key == pygame.K_KP8:
			self.playing_speed = (self.playing_speed + 1) % len(self.PLAY_SPEED)
		elif evt.key == pygame.K_KP2:
			self.playing_speed = (self.playing_speed - 1) % len(self.PLAY_SPEED)

	def read_user_input(self):
		"""
		This is called to read input from the user or the model, depending on which is operating the simulation
		"""
		return []

	def draw_additional_information(self):
		"""
		Draw additional information on the screen for debug and understanding purposes
		"""
		debug_text = ["Frame reward: %.4f" % self.env.compute_reward()]
		if self.step_by_step:
			debug_text.append("Step by step (6: next frame)")
		if self.playing_speed != 0:
			debug_text.append("Speed: %f" % self.PLAY_SPEED[self.playing_speed])

		y = 0
		for line in debug_text:
			text = self.font.render(line, False, (0, 0, 0), (255, 255, 255))
			self.screen.blit(text, (0, y))
			y += text.get_height()

	def next_step(self, user_input):
		"""
		Play a physics step
		"""
		self.env.play_step(user_input, self.dt * self.PLAY_SPEED[self.playing_speed])

	def reset_environment(self):
		"""
		Resets the environment and create a new episode case (Probably called when the environment ended)
		"""
		self.env.new_episode_case()

	def run(self):
		self.env.new_episode_case()
		while self.running:
			for event in pygame.event.get():
				if event.type == pygame.QUIT:
					self.running = False
				elif event.type == pygame.KEYDOWN:
					self.handle_user_action(event)

			self.clock.tick(50)
			if not self.step_by_step or self.play_next_step:
				self.next_step(self.read_user_input())
				self.play_next_step = False

			self.screen.fill(self.fill_color)
			self.env.draw(self.screen)
			self.draw_additional_information()
			pygame.display.flip()

			if self.env.get_state() == BaseEnvironment.STATE_DIED:
				self.reset_environment()
