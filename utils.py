"""
Utility functions for Mario PPO training.
"""
import numpy as np


def compute_policy_health(action_counts: dict, n_actions: int) -> dict:
    """
    Compute policy health statistics from action counts.
    
    Args:
        action_counts: Dictionary mapping action -> count
        n_actions: Total number of possible actions
    
    Returns:
        dict with:
            - action_probs: Array of action probabilities
            - dominant_action: Most frequent action
            - dominant_ratio: Ratio of dominant action (0-1)
            - entropy: Entropy of action distribution
            - is_collapsed: True if policy appears collapsed
    """
    total = sum(action_counts.values())
    if total == 0:
        return {
            'action_probs': np.ones(n_actions) / n_actions,
            'dominant_action': 0,
            'dominant_ratio': 1.0 / n_actions,
            'entropy': np.log(n_actions),
            'is_collapsed': False
        }
    
    # Build probability array
    action_probs = np.array([action_counts.get(i, 0) / total for i in range(n_actions)])
    
    # Find dominant action
    dominant_action = int(np.argmax(action_probs))
    dominant_ratio = float(action_probs[dominant_action])
    
    # Compute entropy: -sum(p * log(p))
    eps = 1e-8
    entropy = float(-np.sum(action_probs * np.log(action_probs + eps)))
    
    # Check for collapse (>85% one action or entropy < 0.3)
    is_collapsed = dominant_ratio > 0.85 or entropy < 0.3
    
    return {
        'action_probs': action_probs,
        'dominant_action': dominant_action,
        'dominant_ratio': dominant_ratio,
        'entropy': entropy,
        'is_collapsed': is_collapsed
    }


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    Returns a function that computes current learning rate based on remaining progress.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(env_id, render_mode=None, use_reward_shaping=True):
    """
    Create a single preprocessed environment.
    Used for vectorized environments.
    
    Args:
        env_id: Environment ID
        render_mode: Render mode (None for training, 'human' for visualization)
        use_reward_shaping: Apply custom reward shaping to prevent policy collapse
    """
    def _init():
        from wrappers import make_mario_env
        env = make_mario_env(
            env_id=env_id,
            actions='simple',  # 7 actions
            skip_frames=4,
            resize_shape=84,
            grayscale=True,
            normalize=True,
            stack_frames=4,
            render_mode=render_mode,
            use_reward_shaping=use_reward_shaping
        )
        return env
    return _init
