import functools
from functools import update_wrapper
import gymnasium as gym
from joblib import Parallel
import numpy as np
import random
from typing import Optional, Tuple


class SlipperyRoomEnv(gym.Env):
    """
    A robot (R) navigates through a grid-world using the actions "up", "down",
    "left", and "right". When reaching the goal (G) state or after 100
    timesteps, an episode ends.
    The robot can run into obstacles (O), which simply bounce the robot back to
    the previous position. The same happens when the robot runs into an outside
    wall.
    Rewards are -1.0 for bumping into obstacles or outside walls, +10.0 for
    reaching the goal and -0.1 otherwise.

    The map looks as follows:
    ----------
    |R    OO |
    |     OO |
    |O       |
    |        |
    |    O   |
    |  OOO   |
    |        |
    |  O    G|
    ----------
    """

    # Row major map (0-based indices). E.g. starting position=(0, 0), going right
    # the robot reaches field (0, 1).
    MAP_ = [
        "R    OO ",
        "     OO ",
        "O       ",
        "        ",
        "    O   ",
        "  OOO   ",
        "        ",
        "  O    G",
    ]

    def __init__(self, config: Optional[dict] = None):
        """Initializes a SlipperyRoomEnv instance.

        Args:
            config (Optional[dict]): An optional config dict. If it contains the
                `is_slippery` key (and value is True), will behave stochastically.
        """
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Discrete(8*8)

        config = config or {}
        # Stochastic actions?
        self.is_slippery = config.get("is_slippery", True)
        # The maximum time steps per episode.
        self.max_timesteps = 50

    def reset(self, seed=None, options=None) -> Tuple[int, dict]:
        """Resets the env and returns the initial observation of the new episode.

        Returns:
            int: The first/initial observation in the new episode.
        """
        if seed is not None:
            np.random.seed(seed + 1)

        self.ts = 0
        self.robot_pos = [0, 0]
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[int, float, bool, bool, dict]:
        """Performs one step in the ongoing episode using `action`.

        Args:
            action (int): The action to take next.

        Returns:
            Tuple[int, float, bool, dict]: Tuple of next observation,
                reward received, boolean done signal, info dict (leave empty!).
        """
        orig_pos = self.robot_pos[:]

        self.ts += 1

        # If env is slippery, change the action to a random one in 30%
        # of the cases.
        if self.is_slippery and random.random() < 0.3:
            action = np.random.randint(0, 3)

        # Up.
        if action == 0:
            self.robot_pos[0] -= 1
        # Right.
        elif action == 1:
            self.robot_pos[1] += 1
        # Down.
        elif action == 2:
            self.robot_pos[0] += 1
        # Left.
        elif action == 3:
            self.robot_pos[1] -= 1
        else:
            raise ValueError("bad action!")

        # Reward/done function.
        reward = -0.1
        truncated = terminated = False
        # Check room boundaries and obstacles.
        if self.robot_pos[0] < 0 or self.robot_pos[0] >= 8 or \
                self.robot_pos[1] < 0 or self.robot_pos[1] >= 8 or \
                self.MAP_[self.robot_pos[0]][self.robot_pos[1]] == "O":
            self.robot_pos = orig_pos
            reward = -1.0
        # Check goal.
        elif self.MAP_[self.robot_pos[0]][self.robot_pos[1]] == "G":
            reward = 10.0
            terminated = True

        # Determine, whether episode is over.
        if terminated is False:
            truncated = self.ts >= self.max_timesteps

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> int:
        """Translates the stored robot position into a discrete (int) value.

        Returns:
            int: The robot's discrete (int) position, given its x/y-pos tuple.
        """
        int_pos = self.robot_pos[0] * 8 + self.robot_pos[1]
        return int_pos


def check_random_state(seed_or_random_state):
    """Turns a seed into a np.random.RandomState instance.

    Args:
        seed (Optional[Union[int, np.random.RandomState]]): If None, return the
            RandomState singleton used by np.random. If int, return a new
            RandomState instance seeded with seed. If seed is already a RandomState
            instance, return it.

    Returns:
        np.random.RandomState
    """
    # Singleton `RandomState` object.
    if seed_or_random_state is None or seed_or_random_state is np.random:
        return np.random.mtrand._rand
    # Int.
    if isinstance(seed_or_random_state, int):
        return np.random.RandomState(seed_or_random_state)
    # Already a RandomState object -> return it.
    if isinstance(seed_or_random_state, np.random.RandomState):
        return seed_or_random_state
    # Otherwise, error.
    raise NotImplementedError


def delayed(function):
    """Decorator used to capture the arguments of a function."""
    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return _FuncWrapper(function), args, kwargs
    return delayed_function


class _FuncWrapper:
    """"Load the global configuration before calling the function."""
    def __init__(self, function):
        self.function = function
        update_wrapper(self, self.function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class SampleCollector:
    def __init__(self, num_jobs=1):
        """Initializes a SampleCollector instance.

        Args:
            num_jobs (int): How many jobs (environments) to run in parallel
                for sample collection.
        """
        self.envs = [SlipperyRoomEnv() for _ in range(num_jobs)]
        self.num_jobs = num_jobs

    def sample(self, num_episodes=10, seed=42):
        """Draws a sample (observations only!) from our n environments in parallel.

        Args:
            num_episodes (int): The number of episodes to sample and return
                (observations only for simplicity) in total.
            seed (int): A seed value to use for deterministic execution.

        Returns:
            np.ndarray: The observation batch collected by running through our
                n envs in parallel.
        """
        random_state = check_random_state(seed)

        assert num_episodes % self.num_jobs == 0
        n = num_episodes // self.num_jobs
        observations = Parallel(n_jobs=self.num_jobs)(delayed(self._run_n_episodes)(
            n, env_idx, random_state) for env_idx in range(len(self.envs)))
        observations = np.concatenate(observations, axis=0)

        return observations

    def _run_n_episodes(self, n, env_idx, random_state: np.random.RandomState):
        num_episodes = 0
        seed = random_state.randint(1e9)
        np.random.seed(seed + 1)

        obs, _ = self.envs[env_idx].reset(seed=seed)

        observations = [obs]
        while True:
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, _ = self.envs[env_idx].step(action)
            if terminated or truncated:
                num_episodes += 1
                if num_episodes >= n:
                    break
                obs, _ = self.envs[env_idx].reset()
            observations.append(obs)

        return np.array(observations)
