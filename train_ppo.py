"""
Train a PPO agent to play Super Mario Bros.

This script uses Stable-Baselines3 PPO with proven hyperparameters
from successful Mario implementations.
"""

import suppress_warnings  # noqa: F401 - must be first to suppress gym warnings

import os
import time
from datetime import datetime

from callbacks import PolicyCollapseCallback
from utils import linear_schedule, make_env, get_level_from_env_id, play


def train(
    env_id="SuperMarioBros-1-1-v0",
    total_timesteps=2_000_000,
    n_envs=8,
    save_dir="./mario_models",
    log_dir="./mario_logs",
    checkpoint_freq=100_000,
    learning_rate=1e-4,
    use_lr_schedule=True,
    resume_from=None,
    ent_coef=0.05,
):
    """
    Train PPO agent on Super Mario Bros.

    Args:
        env_id: Environment ID (use specific level for faster learning)
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save model checkpoints
        log_dir: Directory for tensorboard logs
        checkpoint_freq: Save checkpoint every N steps
        learning_rate: Initial learning rate
        use_lr_schedule: Whether to use linear LR decay
        resume_from: Path to model checkpoint to resume from (optional)
        ent_coef: Entropy coefficient (higher = more exploration)
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
        CallbackList,
    )

    # Extract level name for folder organization
    level = get_level_from_env_id(env_id)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create level-specific directories
    # Structure: mario_models/{level}/checkpoints/{timestamp}/
    #            mario_models/{level}/best/
    #            mario_models/{level}/flag/
    #            mario_logs/{level}/{timestamp}/
    level_model_dir = os.path.join(save_dir, level)
    checkpoint_dir = os.path.join(level_model_dir, "checkpoints", timestamp)
    best_model_dir = os.path.join(level_model_dir, "best")
    flag_model_dir = os.path.join(level_model_dir, "flag")
    run_log_dir = os.path.join(log_dir, level, timestamp)  # Direct path, no suffix
    level_log_dir = os.path.join(log_dir, level)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(flag_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    print("=" * 60)
    print("Super Mario Bros PPO Training")
    print("=" * 60)
    print(f"Environment: {env_id}")
    print(f"Level: {level}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Learning rate: {learning_rate}")
    print(f"LR schedule: {'Linear decay' if use_lr_schedule else 'Constant'}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Best model: {best_model_dir}")
    print(f"Logs: {level_log_dir}")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print("=" * 60)

    # Create vectorized training environment
    print("\nCreating training environments...")
    try:
        # Use SubprocVecEnv for true parallelism (better for multiple envs)
        if n_envs > 1:
            train_env = SubprocVecEnv([make_env(env_id) for i in range(n_envs)])
        else:
            train_env = DummyVecEnv([make_env(env_id)])
        print(f"Training env observation space: {train_env.observation_space}")
        print(f"Training env action space: {train_env.action_space}")
    except Exception as e:
        print(f"SubprocVecEnv failed, falling back to DummyVecEnv: {e}")
        train_env = DummyVecEnv([make_env(env_id) for i in range(n_envs)])

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(env_id)])

    # Setup learning rate
    lr = linear_schedule(learning_rate) if use_lr_schedule else learning_rate

    # Create or load PPO model
    if resume_from:
        print(f"\nLoading model from {resume_from}...")
        model = PPO.load(
            resume_from, env=train_env, tensorboard_log=run_log_dir, device="auto"
        )
        # Update hyperparameters for resumed training
        model.learning_rate = lr
        model.ent_coef = ent_coef
        print(f"Resumed! Previous timesteps: {model.num_timesteps:,}")
        print(f"Entropy coefficient: {model.ent_coef}")
    else:
        print("\nCreating new PPO model...")
        # Policy kwargs to handle normalized images
        policy_kwargs = dict(
            normalize_images=False  # We already normalized to [0, 1]
        )
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=lr,
            n_steps=2048,  # Steps per environment per update
            batch_size=512,  # Larger batch = better GPU utilization
            n_epochs=10,  # Number of epochs per update
            gamma=0.99,  # Discount factor
            gae_lambda=0.95,  # GAE lambda
            clip_range=0.2,  # PPO clip range
            clip_range_vf=None,  # Value function clip (None = no clipping)
            ent_coef=ent_coef,  # Entropy coefficient for exploration
            vf_coef=0.5,  # Value function coefficient
            max_grad_norm=0.5,  # Gradient clipping
            verbose=1,
            tensorboard_log=run_log_dir,
            device="auto",  # Use GPU if available
            policy_kwargs=policy_kwargs,
        )

    print(f"Model on device: {model.device}")
    print(f"Policy: {model.policy}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq // n_envs,  # Divide by n_envs for SubprocVecEnv
        save_path=checkpoint_dir,
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=os.path.join(level_log_dir, "eval"),
        eval_freq=50_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    # Policy collapse detection and auto-recovery callback
    collapse_callback = PolicyCollapseCallback(
        check_freq=50_000,  # Check every 50k steps
        dominant_action_threshold=0.85,  # Trigger if any action > 85%
        entropy_threshold=0.3,  # Trigger if entropy < 0.3
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix="checkpoint",
        n_eval_samples=50,  # Sample 50 batches of observations
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_callback, eval_callback, collapse_callback])

    # Train!
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    print(f"Tensorboard: tensorboard --logdir {level_log_dir}")
    print("Policy collapse detection: ENABLED")
    print("  - Auto-rollback if dominant action > 85%")
    print("  - Auto-rollback if entropy < 0.3")
    print("=" * 60 + "\n")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=not bool(resume_from),
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    elapsed_time = time.time() - start_time

    # Save final model
    final_path = os.path.join(checkpoint_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # Training summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {elapsed_time / 3600:.2f} hours")
    print(f"Steps completed: {model.num_timesteps:,}")
    print(f"Model saved: {final_path}")
    print("=" * 60)

    # Cleanup
    train_env.close()
    eval_env.close()

    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train/Play Super Mario Bros with PPO")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "play"],
        help="Train or play mode",
    )
    parser.add_argument(
        "--env", type=str, default="SuperMarioBros-1-1-v0", help="Environment ID"
    )
    parser.add_argument(
        "--timesteps", type=int, default=2_000_000, help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=16,
        help="Number of parallel environments (default: 16 for RTX 5090)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path for play mode or resume training",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.05,
        help="Entropy coefficient (higher = more exploration, default: 0.05)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Slow down playback for human viewing (play mode only)",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(
            env_id=args.env,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            learning_rate=args.lr,
            resume_from=args.resume,
            ent_coef=args.ent_coef,
        )
    else:
        if args.model is None:
            print("Error: --model required for play mode")
            exit(1)
        play(model_path=args.model, env_id=args.env, slow=args.slow)
