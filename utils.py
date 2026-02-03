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
            "action_probs": np.ones(n_actions) / n_actions,
            "dominant_action": 0,
            "dominant_ratio": 1.0 / n_actions,
            "entropy": np.log(n_actions),
            "is_collapsed": False,
        }

    # Build probability array
    action_probs = np.array(
        [action_counts.get(i, 0) / total for i in range(n_actions)], dtype=np.float32
    )

    # Find dominant action
    dominant_action = int(np.argmax(action_probs))
    dominant_ratio = float(action_probs[dominant_action])

    # Compute entropy: -sum(p * log(p))
    eps = 1e-8
    entropy_value = np.sum(action_probs * np.log(action_probs + eps))
    entropy = float(-1.0 * entropy_value)

    # Check for collapse (>85% one action or entropy < 0.3)
    is_collapsed = dominant_ratio > 0.85 or entropy < 0.3

    return {
        "action_probs": action_probs,
        "dominant_action": dominant_action,
        "dominant_ratio": dominant_ratio,
        "entropy": entropy,
        "is_collapsed": is_collapsed,
    }


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.
    Returns a function that computes current learning rate based on remaining progress.
    """

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


def ent_coef_schedule(initial_value: float, final_value: float = 0.01):
    """
    Linear entropy coefficient schedule.
    Decays from initial_value to final_value over training.
    """

    def func(progress_remaining: float) -> float:
        return final_value + (initial_value - final_value) * progress_remaining

    return func


def make_env(
    env_id,
    render_mode=None,
    use_reward_shaping=True,
):
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
            actions="complex",  # 12 actions (includes down for pipes, up for vines)
            skip_frames=4,
            resize_shape=(128, 120),  # Half native res, preserves aspect ratio
            grayscale=False,  # Keep RGB: color carries information (powerups, level type)
            normalize=False,  # Keep uint8 for memory efficiency, SB3 normalizes on-the-fly
            stack_frames=4,
            render_mode=render_mode,
            use_reward_shaping=use_reward_shaping,
        )
        return env

    return _init


def get_level_from_env_id(env_id: str) -> str:
    """Extract level name from env_id (e.g., 'SuperMarioBros-1-1-v0' -> '1-1')."""
    import re

    match = re.search(r"(\d+-\d+)", env_id)
    if match:
        return match.group(1)
    # For random stages or other envs, use a sanitized name
    return env_id.replace("SuperMarioBros", "").replace("-v0", "").strip("-") or "misc"


def normalize_env_id(env_id: str) -> str:
    """
    Normalize environment ID to full format.

    Supports shorthand notation:
        '1-1' -> 'SuperMarioBros-1-1-v0'
        '1-2' -> 'SuperMarioBros-1-2-v0'
        'SuperMarioBros-1-1-v0' -> 'SuperMarioBros-1-1-v0' (unchanged)

    Args:
        env_id: Environment ID (shorthand or full)

    Returns:
        Full environment ID
    """
    import re

    # Already full format
    if env_id.startswith("SuperMarioBros"):
        return env_id

    # Shorthand format: '1-1', '2-3', etc.
    match = re.match(r"^(\d+-\d+)$", env_id.strip())
    if match:
        return f"SuperMarioBros-{match.group(1)}-v0"

    # Unknown format, return as-is
    return env_id


def normalize_env_ids(env_id: str) -> list[str]:
    """
    Parse and normalize comma-separated environment IDs.

    Args:
        env_id: Single env ID or comma-separated list (e.g., '1-1,1-2')

    Returns:
        List of normalized environment IDs

    Raises:
        ValueError: If no valid environment IDs provided
    """
    env_ids = [e.strip() for e in env_id.split(",")]
    env_ids = [e for e in env_ids if e]  # Filter out empty strings
    if not env_ids:
        raise ValueError(f"No valid environment IDs provided (got {env_id!r})")
    return [normalize_env_id(e) for e in env_ids]


def play(model_path, env_id="1-1", render=True, slow=False):
    """
    Play using a trained model.

    Args:
        model_path: Path to saved model
        env_id: Environment ID or comma-separated list (e.g., '1-1' or '1-1,1-2')
        render: Whether to render the game
        slow: Slow down playback for human viewing
    """
    import time
    from stable_baselines3 import PPO
    from wrappers import make_mario_env

    print(f"Loading model from {model_path}")
    model = PPO.load(model_path)

    # Support multiple environments (comma-separated) with shorthand
    env_ids = normalize_env_ids(env_id)

    render_mode = "human" if render else None

    # Delay between steps for slow mode (in seconds)
    step_delay = 0.05 if slow else 0  # 50ms delay = ~20 FPS

    for env_idx, current_env_id in enumerate(env_ids):
        level_name = get_level_from_env_id(current_env_id)
        if len(env_ids) > 1:
            print(f"\n{'=' * 60}")
            print(f"Level {env_idx + 1}/{len(env_ids)}: {level_name}")
            print("=" * 60)

        env = make_mario_env(
            env_id=current_env_id,
            actions="complex",
            skip_frames=4,
            resize_shape=(128, 120),  # Half native res, preserves aspect ratio
            grayscale=False,  # Keep RGB
            normalize=False,  # Keep uint8, SB3 normalizes on-the-fly
            stack_frames=4,
            render_mode=render_mode,
            use_reward_shaping=False,  # Don't need reward shaping for play mode
        )

        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            # Convert numpy array action to int for JoypadSpace
            action = int(action) if hasattr(action, "__int__") else action.item()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Slow down for human viewing
            if step_delay > 0:
                time.sleep(step_delay)

            if done or truncated:
                break

        flag = info.get("flag_get", False)
        x_pos = info.get("x_pos", 0)
        prefix = f"[{level_name}] " if len(env_ids) > 1 else ""
        print(
            f"{prefix}Reward={total_reward:.2f}, Steps={steps}, "
            f"X_pos={x_pos}, Flag={'Yes!' if flag else 'No'}"
        )

        env.close()
