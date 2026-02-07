"""
Custom wrappers for Super Mario Bros environment.
These handle preprocessing for RL training with PPO.
"""

import suppress_warnings  # noqa: F401 - must be first to suppress gym warnings

from collections import deque

import gym
import numpy as np
from gym import spaces
import cv2


class SkipFrame(gym.Wrapper):
    """
    Return only every `skip`-th frame.
    Repeats the same action for `skip` frames and sums rewards.
    """

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        truncated = False
        obs = None
        info = {}

        for _ in range(self._skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break

        return obs, total_reward, done, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observation to grayscale."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = np.expand_dims(observation, axis=-1)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """Resize observation to (width, height) using cv2.resize convention."""

    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)  # (width, height) for cv2
        else:
            self.shape = tuple(shape)  # (width, height) for cv2

        # cv2.resize uses (width, height), but numpy arrays are (height, width, channels)
        # So output shape is (height, width, channels) = (shape[1], shape[0], channels)
        h, w = self.shape[1], self.shape[0]
        obs_shape = (h, w) + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        if len(observation.shape) == 2:
            observation = np.expand_dims(observation, axis=-1)
        return observation


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observation to [0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=obs_shape, dtype=np.float32
        )

    def observation(self, observation):
        return observation.astype(np.float32) / 255.0


class FrameStack(gym.Wrapper):
    """Stack n_frames consecutive frames."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space
        obs_shape = env.observation_space.shape
        # Stack along the last dimension (channel dimension)
        new_shape = obs_shape[:-1] + (obs_shape[-1] * n_frames,)
        self.observation_space = spaces.Box(
            low=0.0 if env.observation_space.dtype == np.float32 else 0,
            high=1.0 if env.observation_space.dtype == np.float32 else 255,
            shape=new_shape,
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        # Remove seed/options that old gym envs don't support
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        # Initialize frame stack with first observation
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(obs)
        stacked = np.concatenate(self.frames, axis=-1)
        return stacked, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Update frame stack (deque maxlen auto-discards oldest)
        self.frames.append(obs)
        stacked = np.concatenate(self.frames, axis=-1)

        return stacked, reward, done, truncated, info


class TransposeObservation(gym.ObservationWrapper):
    """Transpose observation from (H, W, C) to (C, H, W) for PyTorch CNN."""

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape
        # (H, W, C) -> (C, H, W)
        new_shape = (obs_shape[2], obs_shape[0], obs_shape[1])
        self.observation_space = spaces.Box(
            low=self.observation_space.low.transpose(2, 0, 1),
            high=self.observation_space.high.transpose(2, 0, 1),
            shape=new_shape,
            dtype=self.observation_space.dtype,
        )

    def observation(self, observation):
        # (H, W, C) -> (C, H, W)
        return observation.transpose(2, 0, 1)


class LevelIDWrapper(gym.ObservationWrapper):
    """
    Add level ID as a one-hot vector to the observation.
    Changes observation space to Dict with 'image' and 'level_id'.

    NOTE: Not used - considered "cheating" as it hand-feeds level information
    to the agent instead of letting it learn to distinguish levels from visual
    cues (HUD shows "WORLD 1-1", background colors differ between levels, etc.).
    Kept here for reference/experimentation if needed.
    """

    def __init__(self, env, level_id: int, num_levels: int):
        super().__init__(env)
        self.level_id = level_id
        self.num_levels = num_levels

        self.observation_space = spaces.Dict(
            {
                "image": env.observation_space,
                "level_id": spaces.Box(
                    low=0.0, high=1.0, shape=(num_levels,), dtype=np.float32
                ),
            }
        )

        self._level_one_hot = np.zeros(num_levels, dtype=np.float32)
        self._level_one_hot[level_id] = 1.0

    def observation(self, observation):
        return {
            "image": observation,
            "level_id": self._level_one_hot.copy(),
        }


class CustomRewardWrapper(gym.Wrapper):
    """
    Custom reward shaping to prevent policy collapse.

    Key changes:
    1. Reward forward progress (x_pos increase)
    2. Penalize standing still / moving backward
    3. Bonus for reaching flag
    4. Time pressure to prevent standing still
    5. Score increase reward (coins, enemies, etc.)
    """

    def __init__(
        self,
        env,
        forward_reward_scale=0.1,
        death_penalty=-15,
        stuck_penalty=-0.5,
        flag_bonus=200,
        time_penalty=-0.01,
        score_reward_scale=0.01,
        coin_bonus=5.0,
    ):
        super().__init__(env)
        self._prev_x_pos = 0
        self._prev_time = 400
        self._prev_score = 0
        self._prev_coins = 0
        self._stuck_counter = 0
        self._max_x_pos = 0

        # Reward shaping parameters
        self.forward_reward_scale = forward_reward_scale
        self.death_penalty = death_penalty
        self.stuck_penalty = stuck_penalty
        self.flag_bonus = flag_bonus
        self.time_penalty = time_penalty
        self.score_reward_scale = score_reward_scale
        self.coin_bonus = coin_bonus

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        self._prev_x_pos = info.get("x_pos", 0)
        self._max_x_pos = info.get("x_pos", 0)
        self._prev_time = info.get("time", 400)
        self._prev_score = info.get("score", 0)
        self._prev_coins = info.get("coins", 0)
        self._stuck_counter = 0
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # Get current state
        x_pos = info.get("x_pos", 0)
        time_left = info.get("time", 400)
        flag_get = info.get("flag_get", False)
        score = info.get("score", 0)
        coins = info.get("coins", 0)

        # Calculate shaped reward
        shaped_reward = 0

        # 1. Forward progress reward - only reward NEW maximum territory
        if x_pos > self._max_x_pos:
            progress_reward = (x_pos - self._max_x_pos) * self.forward_reward_scale
            # Apply progressive multiplier: more valuable as Mario progresses
            progress_multiplier = min(1 + (x_pos / 3000), 2.5)
            progress_reward *= progress_multiplier
            shaped_reward += progress_reward
            self._max_x_pos = x_pos

        # 2. Stuck penalty - if Mario hasn't moved in a while
        x_delta = x_pos - self._prev_x_pos
        if x_delta <= 0:
            self._stuck_counter += 1
            if self._stuck_counter > 30:  # Stuck for 30+ frames
                shaped_reward += self.stuck_penalty
        else:
            self._stuck_counter = 0

        # 3. Time pressure - small penalty for time passing
        shaped_reward += self.time_penalty

        # 4. Flag bonus
        if flag_get:
            shaped_reward += self.flag_bonus

        # 5. Death penalty
        if done and not flag_get:
            shaped_reward += self.death_penalty

        # 6. Score increase reward (killing enemies, collecting power-ups, etc.)
        score_delta = score - self._prev_score
        if score_delta > 0:
            shaped_reward += score_delta * self.score_reward_scale

        # 7. Coin bonus
        coin_delta = coins - self._prev_coins
        if coin_delta > 0:
            shaped_reward += coin_delta * self.coin_bonus

        # Combine original reward with shaped reward
        total_reward = reward + shaped_reward

        # Update state
        self._prev_x_pos = x_pos
        self._prev_time = time_left
        self._prev_score = score
        self._prev_coins = coins

        return obs, total_reward, done, truncated, info


def make_mario_env(
    env_id="SuperMarioBros-v0",
    actions="complex",
    skip_frames=4,
    resize_shape=(128, 120),
    grayscale=False,
    normalize=False,
    stack_frames=4,
    render_mode=None,
    use_reward_shaping=True,
):
    """
    Create a preprocessed Super Mario Bros environment.

    Args:
        env_id: Environment ID (e.g., 'SuperMarioBros-v0', 'SuperMarioBros-1-1-v0')
        actions: 'simple' (7 actions), 'right_only' (5 actions), or 'complex' (12 actions)
        skip_frames: Number of frames to skip (repeat action)
        resize_shape: Resize observation to (W, H), None for native 256x240
        grayscale: Convert to grayscale (False keeps RGB)
        normalize: Normalize pixel values to [0, 1] - set False to keep uint8 for memory efficiency
        stack_frames: Number of frames to stack
        render_mode: 'human' for visual rendering, 'rgb_array' for array, None for no render
        use_reward_shaping: Apply custom reward shaping to prevent policy collapse

    Returns:
        Preprocessed environment

    Note:
        Default settings (128x120 RGB, uint8, 4 frames stacked) produce observation
        space of (12, 120, 128) uint8. This keeps full color information while being
        memory efficient (~11GB rollout buffer vs ~180GB for float32 native res).
        SB3 normalizes to float32 on-the-fly when sampling batches (set normalize_images=True
        in policy_kwargs).
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import (
        SIMPLE_MOVEMENT,
        RIGHT_ONLY,
        COMPLEX_MOVEMENT,
    )
    from nes_py.wrappers import JoypadSpace

    # Select action space
    action_map = {
        "simple": SIMPLE_MOVEMENT,
        "right_only": RIGHT_ONLY,
        "complex": COMPLEX_MOVEMENT,
    }
    action_space = action_map.get(actions, SIMPLE_MOVEMENT)

    # Create base environment with compatibility mode for new gym API
    env = gym.make(env_id, apply_api_compatibility=True, render_mode=render_mode)

    # Apply JoypadSpace to reduce action space
    env = JoypadSpace(env, action_space)

    # Apply custom reward shaping BEFORE other wrappers (needs access to info dict)
    if use_reward_shaping:
        env = CustomRewardWrapper(env)

    # Apply preprocessing wrappers
    if skip_frames > 1:
        env = SkipFrame(env, skip=skip_frames)

    if grayscale:
        env = GrayScaleObservation(env)

    if resize_shape:
        env = ResizeObservation(env, shape=resize_shape)

    if normalize:
        env = NormalizeObservation(env)

    if stack_frames > 1:
        env = FrameStack(env, n_frames=stack_frames)

    # Transpose to channel-first format (C, H, W) for PyTorch/SB3
    env = TransposeObservation(env)

    return env


def test_wrappers():
    """Test the preprocessing pipeline."""
    print("Testing preprocessing wrappers...")

    env = make_mario_env(
        env_id="SuperMarioBros-v0",
        actions="complex",
        skip_frames=4,
        resize_shape=(128, 120),  # Half native res
        grayscale=False,  # Keep RGB
        normalize=False,  # Keep uint8
        stack_frames=4,
        render_mode=None,
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min()}, {obs.max()}]")

    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i + 1}: reward={reward:.2f}, done={done}")

        if done:
            env.reset()

    env.close()
    print("Wrapper test completed!")


if __name__ == "__main__":
    test_wrappers()
