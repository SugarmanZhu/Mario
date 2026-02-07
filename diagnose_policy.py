"""
Diagnostic script to analyze policy collapse in the Mario RL agent.

This script helps identify:
1. If the policy has collapsed to a single action
2. Which checkpoints are still healthy
3. Action distribution of the model
"""

import suppress_warnings  # noqa: F401

import os
from glob import glob

from utils import compute_policy_health, get_level_from_env_id, normalize_env_id


def analyze_model_actions(model_path, env, n_steps=500, deterministic=True):
    """
    Analyze the action distribution of a trained model.

    Returns:
        dict: Action statistics including counts, entropy, max_x_pos
    """
    from stable_baselines3 import PPO

    model = PPO.load(model_path)

    obs, info = env.reset()
    action_counts = {}
    total_reward = 0
    max_x_pos = 0

    for _ in range(n_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action) if hasattr(action, "__int__") else action.item()

        action_counts[action] = action_counts.get(action, 0) + 1

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        max_x_pos = max(max_x_pos, info.get("x_pos", 0))

        if done or truncated:
            obs, info = env.reset()

    # Use shared health computation
    health = compute_policy_health(action_counts, env.action_space.n)

    return {
        "action_counts": action_counts,
        "entropy": health["entropy"],
        "max_x_pos": max_x_pos,
        "total_reward": total_reward,
        "dominant_action": health["dominant_action"],
        "dominant_pct": health["dominant_ratio"] * 100,
        "is_collapsed": health["is_collapsed"],
    }


def diagnose_single_model(model_path, render=False, env_id="1-1"):
    """Diagnose a single model for policy collapse."""
    from wrappers import make_mario_env

    # Normalize env_id (support shorthand)
    env_id = normalize_env_id(env_id)

    print(f"\n{'=' * 60}")
    print(f"Diagnosing: {os.path.basename(model_path)}")
    print(f"Environment: {env_id}")
    print("=" * 60)

    render_mode = "human" if render else None
    env = make_mario_env(
        env_id=env_id,
        actions="complex",
        skip_frames=4,
        resize_shape=(128, 120),
        grayscale=False,
        normalize=False,
        stack_frames=4,
        render_mode=render_mode,
        use_reward_shaping=False,
    )

    # Get COMPLEX_MOVEMENT action names
    from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

    action_names = {i: str(COMPLEX_MOVEMENT[i]) for i in range(len(COMPLEX_MOVEMENT))}

    stats = analyze_model_actions(model_path, env, n_steps=500, deterministic=True)

    print(f"\nAction Distribution (deterministic):")
    for action, count in sorted(stats["action_counts"].items()):
        pct = count / 500 * 100
        # Use ASCII characters for progress bar (avoids Unicode issues on Windows)
        bar = "#" * int(pct / 2)
        name = action_names.get(action, f"Action {action}")
        print(f"  {action}: {name:30s} {count:4d} ({pct:5.1f}%) {bar}")

    print(
        f"\nDominant action: {stats['dominant_action']} ({stats['dominant_pct']:.1f}%)"
    )
    print(f"Action entropy: {stats['entropy']:.3f} (higher = more diverse)")
    print(f"Max X position: {stats['max_x_pos']}")
    print(f"Total reward: {stats['total_reward']:.1f}")

    # Diagnosis
    if stats["is_collapsed"]:
        print(f"\n[WARNING] POLICY COLLAPSE DETECTED!")
        print(
            f"   The agent uses action {stats['dominant_action']} ({action_names.get(stats['dominant_action'])}) {stats['dominant_pct']:.1f}% of the time"
        )
        if stats["dominant_action"] == 0:
            print(f"   Action 0 is NOOP - Mario just stands still")
    elif stats["dominant_pct"] > 60:
        print(
            f"\n[WARNING] Policy is becoming biased toward action {stats['dominant_action']}"
        )
    else:
        print(f"\n[OK] Policy appears healthy with diverse actions")

    env.close()
    return stats


def find_healthy_checkpoint(model_dir, max_checkpoints=10, env_id="1-1"):
    """Find the most recent healthy checkpoint before policy collapse."""
    from wrappers import make_mario_env

    # Normalize env_id (support shorthand)
    env_id = normalize_env_id(env_id)

    checkpoints = sorted(glob(os.path.join(model_dir, "**", "*_steps.zip"), recursive=True))

    if not checkpoints:
        print("No checkpoints found!")
        return None

    # Sort by step number
    def get_steps(path):
        name = os.path.basename(path)
        try:
            return int(name.split("_")[-2])
        except (ValueError, IndexError):
            return 0

    checkpoints = sorted(checkpoints, key=get_steps)

    # Test the most recent checkpoints first
    checkpoints_to_test = checkpoints[-max_checkpoints:]

    print(f"\nTesting {len(checkpoints_to_test)} most recent checkpoints...")
    print(f"Environment: {env_id}")

    env = make_mario_env(
        env_id=env_id,
        actions="complex",
        skip_frames=4,
        resize_shape=(128, 120),
        grayscale=False,
        normalize=False,
        stack_frames=4,
        render_mode=None,
        use_reward_shaping=False,
    )

    healthy_checkpoint = None
    best_x_pos = 0

    for ckpt in reversed(checkpoints_to_test):
        print(f"\nTesting: {os.path.basename(ckpt)}")

        try:
            stats = analyze_model_actions(ckpt, env, n_steps=200, deterministic=True)

            print(
                f"  Dominant: {stats['dominant_pct']:.1f}%, X_pos: {stats['max_x_pos']}, Entropy: {stats['entropy']:.3f}"
            )

            if not stats["is_collapsed"] and stats["max_x_pos"] > best_x_pos:
                healthy_checkpoint = ckpt
                best_x_pos = stats["max_x_pos"]
                print(f"  [OK] This looks healthy!")
            elif stats["is_collapsed"]:
                print(f"  [WARNING] Policy collapse detected")
        except Exception as e:
            print(f"  Error: {e}")

    env.close()

    if healthy_checkpoint:
        print(f"\n{'=' * 60}")
        print(f"RECOMMENDED CHECKPOINT: {os.path.basename(healthy_checkpoint)}")
        print(f"Max X position achieved: {best_x_pos}")
        print(f"{'=' * 60}")
    else:
        print(
            f"\n[WARNING] No healthy checkpoints found. You may need to retrain from scratch."
        )

    return healthy_checkpoint


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Diagnose Mario RL policy collapse")
    parser.add_argument(
        "--model", type=str, default=None, help="Path to specific model to diagnose"
    )
    parser.add_argument(
        "--find-healthy",
        action="store_true",
        help="Find the most recent healthy checkpoint",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="./mario_models",
        help="Root model directory (level subdir derived from --env)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game while diagnosing"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="1-1",
        help="Environment ID (e.g., '1-1' or 'SuperMarioBros-1-2-v0')",
    )

    args = parser.parse_args()

    # Build the level-specific model directory that matches the training layout:
    # {model_dir}/{level}/checkpoints/... and {model_dir}/{level}/best/
    level = get_level_from_env_id(normalize_env_id(args.env))
    level_model_dir = os.path.join(args.model_dir, level)

    if args.model:
        diagnose_single_model(args.model, render=args.render, env_id=args.env)
    elif args.find_healthy:
        find_healthy_checkpoint(level_model_dir, env_id=args.env)
    else:
        # Default: diagnose best model
        best_model = os.path.join(level_model_dir, "best", "best_model.zip")
        if os.path.exists(best_model):
            diagnose_single_model(best_model, render=args.render, env_id=args.env)
        else:
            print(
                "No best model found. Use --model to specify a model path or --find-healthy to scan checkpoints."
            )
