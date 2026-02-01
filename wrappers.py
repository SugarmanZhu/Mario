"""
Custom wrappers for Super Mario Bros environment.
These handle preprocessing for RL training with PPO.
"""
import suppress_warnings  # noqa: F401 - must be first to suppress gym warnings

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
    """Resize observation to (shape, shape)."""
    def __init__(self, env, shape=84):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)
        
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def observation(self, observation):
        observation = cv2.resize(
            observation, self.shape, interpolation=cv2.INTER_AREA
        )
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
        self.frames = None
        
        # Update observation space
        obs_shape = env.observation_space.shape
        # Stack along the last dimension (channel dimension)
        new_shape = obs_shape[:-1] + (obs_shape[-1] * n_frames,)
        self.observation_space = spaces.Box(
            low=0.0 if env.observation_space.dtype == np.float32 else 0,
            high=1.0 if env.observation_space.dtype == np.float32 else 255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs):
        # Remove seed/options that old gym envs don't support
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
        
        # Initialize frame stack with copies of first observation
        self.frames = [obs] * self.n_frames
        stacked = np.concatenate(self.frames, axis=-1)
        return stacked, info

    def step(self, action):
        result = self.env.step(action)
        
        if len(result) == 5:
            obs, reward, done, truncated, info = result
        else:
            obs, reward, done, info = result
            truncated = False
        
        # Update frame stack
        self.frames.pop(0)
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
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        # (H, W, C) -> (C, H, W)
        return observation.transpose(2, 0, 1)


def make_mario_env(
    env_id='SuperMarioBros-v0',
    actions='simple',
    skip_frames=4,
    resize_shape=84,
    grayscale=True,
    normalize=True,
    stack_frames=4,
    render_mode=None
):
    """
    Create a preprocessed Super Mario Bros environment.
    
    Args:
        env_id: Environment ID (e.g., 'SuperMarioBros-v0', 'SuperMarioBros-1-1-v0')
        actions: 'simple' (7 actions), 'right_only' (5 actions), or 'complex' (12 actions)
        skip_frames: Number of frames to skip (repeat action)
        resize_shape: Resize observation to (shape, shape)
        grayscale: Convert to grayscale
        normalize: Normalize pixel values to [0, 1]
        stack_frames: Number of frames to stack
        render_mode: 'human' for visual rendering, 'rgb_array' for array, None for no render
    
    Returns:
        Preprocessed environment
    """
    import gym_super_mario_bros
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY, COMPLEX_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    
    # Select action space
    action_map = {
        'simple': SIMPLE_MOVEMENT,
        'right_only': RIGHT_ONLY,
        'complex': COMPLEX_MOVEMENT
    }
    action_space = action_map.get(actions, SIMPLE_MOVEMENT)
    
    # Create base environment with compatibility mode
    env = gym.make(
        env_id,
        apply_api_compatibility=True,
        render_mode=render_mode
    )
    
    # Apply JoypadSpace to reduce action space
    env = JoypadSpace(env, action_space)
    
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
        env_id='SuperMarioBros-v0',
        actions='simple',
        skip_frames=4,
        resize_shape=84,
        grayscale=True,
        normalize=True,
        stack_frames=4,
        render_mode=None
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    
    # Run a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.2f}, done={done}")
        
        if done:
            env.reset()
    
    env.close()
    print("Wrapper test completed!")


if __name__ == "__main__":
    test_wrappers()
