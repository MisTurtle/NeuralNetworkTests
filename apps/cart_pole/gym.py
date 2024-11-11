import pathlib

from apps.cart_pole.environment import *
from apps.utils.gym import BaseGym

from apps.utils.gym_utils import Epsilon, TrainingSettings

MODE_TRAIN = 0
MODE_SHOWCASE = 1

env = CartPoleEnvironment_V3()

gym = BaseGym(
	env,
	TrainingSettings(
		episode_time = 1000, observe = 10000, epsilon = Epsilon.simple(1, 0.05, 0.999),
		save_interval = 1000, save_path=str(pathlib.Path("models/cartpole_{eps}.h5").absolute()),
		train_after=TrainingSettings.TRAIN_AFTER_EPISODES,
		train_policy=lambda gym_stats, episode_stats: gym_stats.get_episode_count() % 10 == 0
	),
	weights = "models/sp_31000.h5"
)
while True:
	if gym.step() == BaseGym.RESULT_GYM_STOPPED:
		break
