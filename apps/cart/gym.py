import pathlib

from apps.cart.environment import CartEnvironment_V2
from apps.utils.gym import BaseGym

import os.path

from apps.utils.gym_utils import TrainingSettings, Epsilon, ReplayBuffer
from apps.utils.learning_modes import DQN

MODE_TRAIN = 0
MODE_SHOWCASE = 1

# Create output dir
if not os.path.exists("models"):
	os.mkdir("models")

env = CartEnvironment_V2()

gym = BaseGym(
	env,
	DQN(env, ReplayBuffer(5000, 128)),
	TrainingSettings(
		episode_time=400, observe=10000, epsilon=Epsilon.simple(1, 0.05, 0.999),
		save_interval=200, save_path=str(pathlib.Path("models/cart_{eps}.h5").absolute()),
		train_after=TrainingSettings.TRAIN_AFTER_EPISODES,
		train_policy=lambda gym_stats, episode_stats: gym_stats.get_episode_count() % 10 == 0
	),
	weights=None
)
while True:
	if gym.step() == BaseGym.RESULT_GYM_STOPPED:
		break
