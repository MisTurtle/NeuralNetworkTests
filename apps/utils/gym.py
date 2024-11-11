import pathlib

import tensorflow as tf

from apps.utils import gym_utils
from apps.utils.environment import BaseEnvironment
from apps.utils.gym_utils import *
from apps.utils.learning_modes import LearningMode


class BaseGym:

    RESULT_NOTHING_NEW = 0  # Nothing really happened
    RESULT_EPISODE_TERMINATED = 1  # Episode was terminated (probably because the agent died)
    RESULT_EPISODE_TIMED_OUT = 2  # Episode was terminated due to time out
    RESULT_GYM_STOPPED = 3  # Gym training has finished

    def __init__(self, env: BaseEnvironment, mode: LearningMode, settings: TrainingSettings, weights: str = None, **kwargs):
        self.env = env
        self.mode = mode
        self.settings = settings
        self.epsilon = self.settings.epsilon
        self.episode_stats, self.gym_stats = StatisticsContainer(env.get_environment_name()), GymStatistics(env.get_environment_name())
        self.initialized = False

        if weights is not None:
            self.mode.load(weights)
        if kwargs.get('summary', True):
            self.mode.get_model().summary()

        self._init()

    def get_env(self) -> BaseEnvironment:
        return self.env

    def observing(self) -> bool:
        return self.gym_stats.ticks_count < self.settings.observe

    def concluded(self) -> bool:
        return self.gym_stats.episode_count >= self.settings.episodes

    def _init(self):
        """
        Gets the gym ready to run by initializing the environment and various data like the input shape
        """
        self.env.setup_environment()
        self.env.new_episode_case()
        self.initialized = True

    def next_episode(self):
        eps_id = self.gym_stats.episode_count + 1
        if eps_id % 10 == 1:
            print(self.settings.episode_end_header(eps_id) + " " + str(self.episode_stats))
        if self.settings.should_save_model(eps_id):
            path = self.settings.get_save_path(eps_id)
            self.mode.save(path)
            print("> Model saved to " + str(pathlib.Path(path).absolute()))

        if not self.observing():
            self.gym_stats.on_episode_ends(self.episode_stats)
            self.epsilon.decay()

        self.episode_stats.reset()
        self.env.new_episode_case()

    def get_action(self, state):
        if self.epsilon.decide_greedy():  # Epsilon-greedy policy is here but might be more appropriate as a separate Mode abstract class
            return self.env.random_input()
        return self.mode.get_action(state)

    def step(self):
        """
        Performs one step in the training, and return a code for the step result
        This currently uses Deep Q-Leaning, but I'll be working on a polymorphic version now
        """
        if self.settings.is_timed_out(self.episode_stats.ticks_count) or self.env.get_state() == BaseEnvironment.STATE_DIED:
            self.next_episode()

        ends = self.mode.step()
        ends |= self.settings.is_timed_out(self.episode_stats.ticks_count)

        self.episode_stats.tick(gym_utils.TIME_STEP)
        self.gym_stats.tick(gym_utils.TIME_STEP)
        self.episode_stats.reward_history.append(self.env.compute_reward())

        if not self.observing() and (self.settings.train_after == self.settings.TRAIN_AFTER_TIME_STEPS or ends) and self.settings.should_train(self.gym_stats, self.episode_stats):
            self.mode.train()

        if self.gym_stats.ticks_count == self.settings.observe:
            print("Observation done. Starting training.")

        return self.get_return_code()

    def get_return_code(self) -> int:
        if self.settings.is_gym_complete(self.gym_stats.episode_count, self.gym_stats.get_real_duration()):
            return BaseGym.RESULT_GYM_STOPPED

        if self.settings.is_timed_out(self.episode_stats.ticks_count):
            return BaseGym.RESULT_EPISODE_TIMED_OUT

        if self.env.get_state() == BaseEnvironment.STATE_DIED:
            return BaseGym.RESULT_EPISODE_TERMINATED

        return BaseGym.RESULT_NOTHING_NEW
