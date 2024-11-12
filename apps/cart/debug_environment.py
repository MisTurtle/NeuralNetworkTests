import pygame.key

from apps.cart.environment import CartEnvironment
from apps.utils.debug import SimplePygameDebugEnvironment


class CartDebugEnvironment(SimplePygameDebugEnvironment):

	def __init__(self, env: CartEnvironment):
		super().__init__(env)
		self.input_history = [0, 0]  # Left taps, right taps

	def reset_environment(self):
		super().reset_environment()
		self.input_history = [0, 0]

	def read_user_input(self):
		keys = pygame.key.get_pressed()
		return 0 if keys[pygame.K_LEFT] else (1 if keys[pygame.K_RIGHT] else None)

	def next_step(self, user_input):
		super().next_step(user_input)
		if user_input == 0:
			self.input_history[0] += 1
		elif user_input == 1:
			self.input_history[1] += 1

	def draw_additional_information(self):
		super().draw_additional_information()
		env: CartEnvironment = self.get_env()
		pygame.draw.circle(self.screen, (0, 255, 0), env.target, env.target_radius)

		lines = []
		velocity = env.cart_body.velocity.x
		lines.append("Cart Rel. Position : %.4f" % (env.cart_body.position.x / env.get_environment_size()[0] - 0.5))
		lines.append("Cart Velocity : %.4f" % velocity)
		lines.append("Input Stats : {Left = %d; Right = %d}" % (self.input_history[0], self.input_history[1]))

		y = env.get_environment_size()[1] - self.font.get_height()
		for line in lines:
			text = self.font.render(line, False, (0, 0, 0), (255, 255, 255))
			self.screen.blit(text, (0, y))
			y -= text.get_height()
