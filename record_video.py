"""
Record gameplay video/GIF of a trained Mario agent.

Usage:
    python record_video.py --model mario_models/flag/1-1-v0.zip --output assets/demo-1-1.gif --fps 30
"""

import suppress_warnings  # noqa: F401 - must be first

import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from wrappers import make_mario_env


def record_episode(
    model_path: str, env_id: str = "SuperMarioBros-1-1-v0", max_steps: int = 10000
) -> tuple[list[np.ndarray], dict]:
    """
    Record a single episode of gameplay.

    Args:
        model_path: Path to the trained model
        env_id: Environment ID
        max_steps: Maximum steps per episode

    Returns:
        Tuple of (frames, info) where frames is a list of RGB arrays
    """
    import gym
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    from stable_baselines3 import PPO

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create the wrapped environment for the model (no rendering)
    env = make_mario_env(
        env_id=env_id,
        actions="simple",
        skip_frames=4,
        resize_shape=84,
        grayscale=True,
        normalize=True,
        stack_frames=4,
        render_mode=None,
        use_reward_shaping=False,
    )

    # Create a separate raw environment just for capturing frames
    raw_env = gym.make(env_id, apply_api_compatibility=True, render_mode="rgb_array")
    raw_env = JoypadSpace(raw_env, SIMPLE_MOVEMENT)

    frames = []
    obs, info = env.reset()
    raw_env.reset()
    done = False
    total_reward = 0
    steps = 0

    print("Recording episode...")

    while not done and steps < max_steps:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action = int(action) if hasattr(action, "__int__") else action.item()

        # Step both environments with the same action
        # We need to repeat the action 4 times for raw_env since wrapped env uses skip_frames=4
        for _ in range(4):
            frame = raw_env.render()
            if frame is not None:
                frames.append(frame.copy())
            raw_obs, _, raw_done, _, raw_info = raw_env.step(action)
            if raw_done:
                break

        # Step the wrapped environment once (it handles frame skip internally)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

        if done or truncated:
            break

    env.close()
    raw_env.close()

    result_info = {
        "total_reward": total_reward,
        "steps": steps,
        "flag_get": info.get("flag_get", False),
        "x_pos": info.get("x_pos", 0),
        "time": info.get("time", 0),
    }

    print(f"Episode complete!")
    print(f"  Steps: {steps}")
    print(f"  Reward: {total_reward:.2f}")
    print(f"  X Position: {result_info['x_pos']}")
    print(f"  Flag Get: {'Yes!' if result_info['flag_get'] else 'No'}")
    print(f"  Frames captured: {len(frames)}")

    return frames, result_info


def save_gif(
    frames: list[np.ndarray],
    output_path: str,
    fps: int = 30,
    optimize: bool = True,
    loop: int = 0,
):
    """
    Save frames as an animated GIF.

    Args:
        frames: List of RGB numpy arrays
        output_path: Output file path
        fps: Frames per second
        optimize: Optimize GIF size
        loop: Number of loops (0 = infinite)
    """
    if not frames:
        print("No frames to save!")
        return

    print(f"Saving GIF to {output_path}...")

    # Convert numpy arrays to PIL Images
    pil_frames = [Image.fromarray(frame) for frame in frames]

    # Calculate duration in milliseconds
    duration = int(1000 / fps)

    # Save as GIF
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=loop,
        optimize=optimize,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"GIF saved! Size: {file_size:.2f} MB")


def save_mp4(frames: list[np.ndarray], output_path: str, fps: int = 30):
    """
    Save frames as an MP4 video.

    Args:
        frames: List of RGB numpy arrays
        output_path: Output file path
        fps: Frames per second
    """
    try:
        import cv2
    except ImportError:
        print(
            "OpenCV (cv2) is required for MP4 output. Install with: pip install opencv-python"
        )
        return

    if not frames:
        print("No frames to save!")
        return

    print(f"Saving MP4 to {output_path}...")

    height, width = frames[0].shape[:2]

    # Use mp4v codec (VideoWriter.fourcc is better recognized by IDEs than VideoWriter_fourcc)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        # Convert RGB to BGR for OpenCV
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(bgr_frame)

    out.release()

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"MP4 saved! Size: {file_size:.2f} MB")


def save_webp(
    frames: list[np.ndarray], output_path: str, fps: int = 30, quality: int = 80
):
    """
    Save frames as an animated WebP (smaller than GIF, better quality).

    Args:
        frames: List of RGB numpy arrays
        output_path: Output file path
        fps: Frames per second
        quality: WebP quality (0-100)
    """
    if not frames:
        print("No frames to save!")
        return

    print(f"Saving WebP to {output_path}...")

    pil_frames = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)

    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        quality=quality,
    )

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"WebP saved! Size: {file_size:.2f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Record gameplay video/GIF of a trained Mario agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.gif
  python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.mp4 --fps 30
  python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.webp --quality 90
        """,
    )

    parser.add_argument(
        "--model", type=str, required=True, help="Path to the trained model (.zip file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="demo.gif",
        help="Output file path (supports .gif, .mp4, .webp)",
    )
    parser.add_argument(
        "--env", type=str, default="SuperMarioBros-1-1-v0", help="Environment ID"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Frames per second (default: 30)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=10000,
        help="Maximum steps per episode (default: 10000)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=80,
        help="Quality for WebP output (0-100, default: 80)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record (default: 1)",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Record multiple episodes and keep the best (flag_get=True)",
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return 1

    # Determine output format
    output_path = Path(args.output)
    output_format = output_path.suffix.lower()

    if output_format not in [".gif", ".mp4", ".webp"]:
        print(f"Error: Unsupported output format: {output_format}")
        print("Supported formats: .gif, .mp4, .webp")
        return 1

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_frames = None
    best_info = None

    for episode in range(args.episodes):
        if args.episodes > 1:
            print(f"\n=== Episode {episode + 1}/{args.episodes} ===")

        frames, info = record_episode(
            model_path=args.model, env_id=args.env, max_steps=args.max_steps
        )

        # Keep best episode (prioritize flag_get, then x_pos)
        if best_frames is None:
            best_frames = frames
            best_info = info
        elif args.best:
            if info["flag_get"] and not best_info["flag_get"]:
                best_frames = frames
                best_info = info
            elif (
                info["flag_get"] == best_info["flag_get"]
                and info["x_pos"] > best_info["x_pos"]
            ):
                best_frames = frames
                best_info = info

    if args.episodes > 1:
        print(f"\n=== Best Episode ===")
        print(f"  X Position: {best_info['x_pos']}")
        print(f"  Flag Get: {'Yes!' if best_info['flag_get'] else 'No'}")

    # Save output
    print(f"\n=== Saving Output ===")

    if output_format == ".gif":
        save_gif(best_frames, str(output_path), fps=args.fps)
    elif output_format == ".mp4":
        save_mp4(best_frames, str(output_path), fps=args.fps)
    elif output_format == ".webp":
        save_webp(best_frames, str(output_path), fps=args.fps, quality=args.quality)

    print(f"\nDone! Output saved to: {output_path}")
    return 0


if __name__ == "__main__":
    exit(main())
