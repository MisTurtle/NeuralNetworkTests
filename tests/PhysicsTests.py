import pygame
import pymunk
import pymunk.pygame_util
import math

# Initialize Pygame and set up the screen
pygame.init()
screen = pygame.display.set_mode((800, 600))
clock = pygame.time.Clock()
running = True

# Initialize Pymunk space
space = pymunk.Space()
space.gravity = (0, -981)
pymunk.pygame_util.positive_y_is_up = True

# Create a ball
ball_radius = 20
ball_mass = 10
ball_moment = pymunk.moment_for_circle(ball_mass, 0, ball_radius)
ball_body = pymunk.Body(ball_mass, ball_moment)
ball_body.position = (150, 500)  # Starting position of the ball
ball_shape = pymunk.Circle(ball_body, ball_radius)
ball_shape.elasticity = 0.8
space.add(ball_body, ball_shape)

# Create a curved floor
curve_segments = []
floor_x_start = 50
floor_x_end = 750
floor_y_base = 100
num_segments = 50

for i in range(num_segments):
	x1 = floor_x_start + i * ((floor_x_end - floor_x_start) / num_segments)
	x2 = floor_x_start + (i + 1) * ((floor_x_end - floor_x_start) / num_segments)

	# Create a sine wave for the curved floor
	y1 = floor_y_base + 50 * math.sin(i * 0.3)
	y2 = floor_y_base + 50 * math.sin((i + 1) * 0.3)

	segment_shape = pymunk.Segment(space.static_body, (x1, y1), (x2, y2), 5)
	segment_shape.elasticity = 0.9
	curve_segments.append(segment_shape)
	space.add(segment_shape)

# Pygame drawing options
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Main loop
while running:
	# Event handling
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			running = False

	# Step the physics simulation
	space.step(1 / 60)

	# Clear the screen
	screen.fill((255, 255, 255))  # White background

	# Draw the Pymunk objects
	space.debug_draw(draw_options)

	# Update the display
	pygame.display.flip()

	# Control frame rate
	clock.tick(60)

# Clean up
pygame.quit()
