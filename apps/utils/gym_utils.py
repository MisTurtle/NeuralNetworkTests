import time
import random
from collections import deque

import numpy as np

TIME_STEP = 0.02


class ReplayBuffer:

    def __init__(self, memory_size: int, batch_size: int = -1):
        self.replays = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, ends):
        self.replays.append((state, action, reward, next_state, ends))

    def sample(self, size: int = -1):
        if size < 0:
            size = self.batch_size
        if size < 0 or len(self.replays) < size:
            return None
        return random.sample(self.replays, size)

    def np_sample(self, size: int = -1):
        s = self.sample(size)
        if s is None:
            return None
        return map(lambda x: np.array(x), zip(*self.sample(size)))


class Epsilon:
    """
    Epsilon-greedy epsilon object
    This can be upgraded later on for more complex epsilon-greedy exploration
    """
    @staticmethod
    def default():
        return Epsilon(1, 0.01, 0.99)

    @staticmethod
    def simple(start_value, end_value, decay_rate):
        return Epsilon(start_value, end_value, decay_rate)

    @staticmethod
    def none():
        return Epsilon(0, 0, 0)

    @staticmethod
    def constant(val: float):
        return Epsilon(val, val, 1)

    def __init__(self, start_value, end_value, decay_rate):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_rate = decay_rate
        self.epsilon = start_value

    def get(self) -> float:
        return self.epsilon

    def decay(self):
        if self.epsilon > self.end_value:
            self.epsilon *= self.decay_rate

    def decide_greedy(self) -> bool:
        return random.random() < self.epsilon

    def reset(self):
        self.epsilon = self.start_value


class StatisticsContainer:

    def __init__(self, container_name: str):
        self.container_name = container_name
        self.simulated_time = 0
        self.ticks_count = 0
        self.real_start_time = time.time()
        self.reward_history = []
        self.last_action = None

    def tick(self, dt: float):
        self.simulated_time += dt
        self.ticks_count += 1

    def reset(self):
        self.simulated_time = 0
        self.ticks_count = 0
        self.real_start_time = time.time()
        self.reward_history.clear()

    def get_ticks_count(self) -> int:
        return self.ticks_count

    def get_real_duration(self):
        return time.time() - self.real_start_time

    def get_simulated_duration(self):
        return self.simulated_time

    def get_average_reward(self):
        if len(self.reward_history) == 0:
            return 0
        return sum(self.reward_history) / len(self.reward_history)

    def get_final_reward(self):
        return sum(self.reward_history)

    def get_last_action(self):
        return self.last_action

    def __str__(self):
        return "%s: Duration: %.2fs, Simulated Time: %.2fs, Total Reward: %.4f, Avg Reward: %.4f" % (
            self.container_name,
            self.get_real_duration(),
            self.get_simulated_duration(),
            self.get_final_reward(),
            self.get_average_reward()
        )


class GymStatistics(StatisticsContainer):

    def __init__(self, container_name: str):
        super().__init__(container_name)
        self.episode_count = 0

    def get_episode_count(self) -> int:
        return self.episode_count

    def on_episode_ends(self, stats: StatisticsContainer):
        self.episode_count += 1
        self.reward_history.append(stats.get_final_reward())


class TrainingSettings:
    """
    A structure regrouping various training parameters for a Gym training session
    Keyword arguments:

        episodes => Max number of episodes to run during the session
        episode_time => Time after witch an episode is forcefully terminated (It could be a good or bad thing, depending on the context)
        max_real_duration => How much time to allocate to this particular training session (terminate the gym session on overshoot)
        observe => How many time steps are performed before starting the training process (Observation phase)
        epsilon => Settings for the epsilon greedy method. Epsilon.none() is the default
        save_interval => After how many episodes to save the current model as a file
        save_path => Where to save the model ({eps} will be replaced by the current episode id, e.g. eps_{eps}.h5)
    """

    TRAIN_AFTER_TIME_STEPS = 0
    TRAIN_AFTER_EPISODES = 1

    def __init__(self, **kwargs):
        self.episodes = kwargs.get('episodes', 0)
        self.episode_time = kwargs.get('episode_time', 0)
        self.max_real_duration = kwargs.get('max_real_duration', 0)
        self.observe = kwargs.get('observe', 0)
        self.epsilon = kwargs.get('epsilon', Epsilon.none())
        self.save_model_interval = kwargs.get('save_interval', 25)
        self.save_model_path = kwargs.get('save_path', "models/eps_{eps}.h5")
        self.train_after = kwargs.get('train_after', self.TRAIN_AFTER_TIME_STEPS)
        self.train_policy = kwargs.get('train_policy', lambda gym_stats, episode_stats: gym_stats.get_ticks_count() % 10 == 0)

    def is_timed_out(self, ticks_count: int) -> bool:
        return 0 < self.episode_time <= ticks_count

    def is_gym_complete(self, episode_count: int, real_time: float) -> bool:
        return (episode_count >= self.episodes > 0) or (real_time >= self.max_real_duration > 0)

    def should_save_model(self, episode_number: int):
        if self.save_model_interval <= 0:
            return False
        return episode_number > 0 and episode_number % self.save_model_interval == 0

    def should_train(self, gym_stats: GymStatistics, eps_stats: StatisticsContainer):
        return self.train_policy(gym_stats, eps_stats)

    def get_save_path(self, episode_number: int):
        return self.save_model_path.replace("{eps}", str(episode_number))

    def episode_end_header(self, episode_number: int) -> str:
        epsilon_text = ""
        if self.epsilon.get() > 0.0001:
            epsilon_text = " [Eps: %.4f]" % self.epsilon.get()
        if self.episodes > 0:
            return f"[{episode_number}/{self.episodes}]" + epsilon_text
        return f"[Episode {episode_number}]" + epsilon_text
