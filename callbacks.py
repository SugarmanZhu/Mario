"""
Custom callbacks for PPO training.
"""

import os
import glob
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from utils import compute_policy_health


class EntropyDecayCallback(BaseCallback):
    """
    Callback that decays the entropy coefficient during training.

    SB3 doesn't support schedule functions for ent_coef (only for learning_rate),
    so we manually decay it using this callback.
    """

    def __init__(
        self,
        initial_ent_coef: float = 0.08,
        final_ent_coef: float = 0.01,
        total_timesteps: int = 20_000_000,
        verbose: int = 1,
    ):
        """
        Args:
            initial_ent_coef: Starting entropy coefficient
            final_ent_coef: Final entropy coefficient at end of training
            total_timesteps: Total expected training timesteps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.total_timesteps = total_timesteps

    def _on_step(self) -> bool:
        """Called after each step - update entropy coefficient."""
        # Calculate progress (0 to 1)
        progress = min(1.0, self.num_timesteps / self.total_timesteps)

        # Linear decay from initial to final
        new_ent_coef = self.final_ent_coef + (
            self.initial_ent_coef - self.final_ent_coef
        ) * (1 - progress)

        # Update the model's entropy coefficient
        self.model.ent_coef = new_ent_coef

        # Log to tensorboard periodically
        if self.n_calls % 10000 == 0:
            self.logger.record("train/ent_coef", new_ent_coef)
            if self.verbose > 1:
                print(
                    f"Entropy coef: {new_ent_coef:.4f} (progress: {progress * 100:.1f}%)"
                )

        return True


class PolicyCollapseCallback(BaseCallback):
    """
    Callback that monitors for policy collapse during training.

    Policy collapse is detected when:
    - A single action dominates (>85% of actions)
    - Entropy drops too low (<0.3)

    When collapse is detected, automatically rolls back to the last healthy checkpoint.
    """

    def __init__(
        self,
        check_freq: int = 10_000,
        dominant_action_threshold: float = 0.85,
        entropy_threshold: float = 0.3,
        checkpoint_dir: str = "./mario_models",
        checkpoint_prefix: str = "mario_ppo_",
        n_eval_samples: int = 100,
        verbose: int = 1,
    ):
        """
        Args:
            check_freq: Check for collapse every N steps
            dominant_action_threshold: Trigger if any action > this % (0.85 = 85%)
            entropy_threshold: Trigger if entropy < this value
            checkpoint_dir: Directory where checkpoints are saved
            checkpoint_prefix: Prefix for checkpoint files
            n_eval_samples: Number of observations to sample for checking
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq
        self.dominant_action_threshold = dominant_action_threshold
        self.entropy_threshold = entropy_threshold
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = checkpoint_prefix
        self.n_eval_samples = n_eval_samples

        # Track last healthy checkpoint
        self.last_healthy_checkpoint: Optional[str] = None
        self.last_healthy_step: int = 0
        self.collapse_count: int = 0
        self.recovery_count: int = 0

    def _find_latest_checkpoint(self) -> Optional[str]:
        """Find the most recent checkpoint file."""
        pattern = os.path.join(self.checkpoint_dir, f"{self.checkpoint_prefix}*.zip")
        checkpoints = glob.glob(pattern)
        if not checkpoints:
            return None
        # Sort by modification time, get newest
        return max(checkpoints, key=os.path.getmtime)

    def _find_checkpoint_before_step(self, step: int) -> Optional[str]:
        """Find checkpoint from before the given step."""
        pattern = os.path.join(
            self.checkpoint_dir, f"{self.checkpoint_prefix}*_steps.zip"
        )
        checkpoints = glob.glob(pattern)

        valid_checkpoints = []
        for cp in checkpoints:
            # Extract step number from filename like "mario_ppo_20260201_111553_3200000_steps.zip"
            try:
                basename = os.path.basename(cp)
                # Find the step number (second-to-last part before _steps.zip)
                parts = basename.replace("_steps.zip", "").split("_")
                cp_step = int(parts[-1])
                if cp_step < step:
                    valid_checkpoints.append((cp, cp_step))
            except (ValueError, IndexError):
                continue

        if not valid_checkpoints:
            return None

        # Return the checkpoint with the highest step count that's still before the collapse
        valid_checkpoints.sort(key=lambda x: x[1], reverse=True)
        return valid_checkpoints[0][0]

    def _collect_action_counts(self) -> dict:
        """
        Collect action counts by sampling from the policy's action distribution.

        NOTE: We do NOT interact with the training environment here to avoid
        disrupting SubprocVecEnv and causing deadlocks. Instead, we sample
        random observations and query the policy.

        Returns:
            dict mapping action -> count
        """
        action_counts = {}
        n_envs = self.training_env.num_envs
        obs_shape = self.training_env.observation_space.shape

        # Generate random observations (normalized pixel values)
        # This tests if the policy has collapsed to always choosing one action
        for _ in range(self.n_eval_samples):
            # Create random observations (batch of n_envs observations)
            random_obs = np.random.rand(n_envs, *obs_shape).astype(np.float32)

            # Get action from policy
            action, _ = self.model.predict(random_obs, deterministic=False)

            # Count actions
            for a in action:
                a = int(a)
                action_counts[a] = action_counts.get(a, 0) + 1

        return action_counts

    def _on_step(self) -> bool:
        """Called after each step."""
        if self.n_calls % self.check_freq != 0:
            return True

        try:
            # Collect actions and compute health stats
            action_counts = self._collect_action_counts()
            n_actions = self.training_env.action_space.n
            health = compute_policy_health(action_counts, n_actions)

            dominant_action = health["dominant_action"]
            dominant_ratio = health["dominant_ratio"]
            entropy = health["entropy"]

            # Log to tensorboard
            self.logger.record("policy_health/dominant_action", dominant_action)
            self.logger.record("policy_health/dominant_ratio", dominant_ratio)
            self.logger.record("policy_health/entropy", entropy)
            self.logger.record("policy_health/collapse_count", self.collapse_count)
            self.logger.record("policy_health/recovery_count", self.recovery_count)

            # Check for collapse using our thresholds
            is_collapsed = (
                dominant_ratio > self.dominant_action_threshold
                or entropy < self.entropy_threshold
            )

            if is_collapsed:
                self.collapse_count += 1

                if self.verbose > 0:
                    print("\n" + "!" * 60)
                    print("WARNING: POLICY COLLAPSE DETECTED!")
                    print(f"    Step: {self.num_timesteps:,}")
                    print(
                        f"    Dominant action: {dominant_action} ({dominant_ratio * 100:.1f}%)"
                    )
                    print(f"    Entropy: {entropy:.4f}")
                    print("!" * 60)

                # Try to recover from last healthy checkpoint
                if self.last_healthy_checkpoint and os.path.exists(
                    self.last_healthy_checkpoint
                ):
                    if self.verbose > 0:
                        print(f"Rolling back to: {self.last_healthy_checkpoint}")
                        print(f"   (from step {self.last_healthy_step:,})")

                    # Load the healthy checkpoint
                    self.model.set_parameters(self.last_healthy_checkpoint)
                    self.recovery_count += 1

                    if self.verbose > 0:
                        print("Recovery complete! Continuing training...")
                        print("!" * 60 + "\n")
                else:
                    # Try to find a checkpoint from before current step
                    fallback = self._find_checkpoint_before_step(self.num_timesteps)
                    if fallback:
                        if self.verbose > 0:
                            print(f"No healthy checkpoint tracked. Using: {fallback}")
                        self.model.set_parameters(fallback)
                        self.recovery_count += 1
                        if self.verbose > 0:
                            print("Recovery complete! Continuing training...")
                            print("!" * 60 + "\n")
                    else:
                        if self.verbose > 0:
                            print("ERROR: No checkpoint available for recovery!")
                            print("   Training will continue but may not recover.")
                            print("!" * 60 + "\n")
            else:
                # Policy is healthy - update our reference
                current_checkpoint = self._find_latest_checkpoint()
                if (
                    current_checkpoint
                    and current_checkpoint != self.last_healthy_checkpoint
                ):
                    self.last_healthy_checkpoint = current_checkpoint
                    self.last_healthy_step = self.num_timesteps

                    if self.verbose > 1:
                        print(f"Healthy checkpoint updated: {current_checkpoint}")

        except Exception as e:
            if self.verbose > 0:
                print(f"Warning: PolicyCollapseCallback check failed: {e}")

        return True


class ProgressTrackingCallback(BaseCallback):
    """
    Callback that tracks and logs episode-level progress metrics to TensorBoard.

    Tracks:
    - max_x_pos: Maximum x position reached in the level
    - flag_get: Whether the level was completed (flag reached)
    - episode_length: Number of steps taken in the episode

    Uses a rolling window of the last 100 episodes to compute statistics.
    """

    def __init__(self, check_freq: int = 1_000, verbose: int = 1):
        """
        Args:
            check_freq: Log statistics every N steps
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.check_freq = check_freq

        # Rolling window storage (last 100 episodes)
        self.max_window_size = 100
        self.episode_max_x_pos: list = []
        self.episode_flag_get: list = []
        self.episode_lengths: list = []

        # Track current episode per environment (initialized on first step)
        self.current_episode_max_x: Optional[np.ndarray] = None
        self.current_episode_length: Optional[np.ndarray] = None

    def _init_tracking(self) -> None:
        """Initialize tracking arrays for each environment."""
        n_envs = self.training_env.num_envs
        self.current_episode_max_x = np.zeros(n_envs, dtype=np.float32)
        self.current_episode_length = np.zeros(n_envs, dtype=np.int32)

    def _on_step(self) -> bool:
        """Called after each step."""
        # Initialize tracking on first step
        if self.current_episode_max_x is None:
            self._init_tracking()

        # Track episode statistics from the environment
        # We access the last dones and infos from the training environment's step
        if hasattr(self.training_env, "buf_dones") and hasattr(
            self.training_env, "buf_infos"
        ):
            dones = self.training_env.buf_dones
            infos = self.training_env.buf_infos
        else:
            # Fallback: try to get from wrapped env
            dones = getattr(
                self.training_env,
                "dones",
                np.zeros(self.training_env.num_envs, dtype=bool),
            )
            infos = getattr(
                self.training_env, "infos", [{}] * self.training_env.num_envs
            )

        # Update tracking for each environment
        for i in range(self.training_env.num_envs):
            info = infos[i] if i < len(infos) else {}

            # Track max x position
            if (
                isinstance(info, dict)
                and "x_pos" in info
                and self.current_episode_max_x is not None
            ):
                self.current_episode_max_x[i] = max(
                    self.current_episode_max_x[i], float(info.get("x_pos", 0))
                )

            # Increment step counter
            if self.current_episode_length is not None:
                self.current_episode_length[i] += 1

            # Check if episode is done
            if i < len(dones) and dones[i]:
                # Episode ended - record stats
                flag_get = (
                    bool(info.get("flag_get", False))
                    if isinstance(info, dict)
                    else False
                )

                if (
                    self.current_episode_max_x is not None
                    and self.current_episode_length is not None
                ):
                    self.episode_max_x_pos.append(float(self.current_episode_max_x[i]))
                    self.episode_flag_get.append(1 if flag_get else 0)
                    self.episode_lengths.append(int(self.current_episode_length[i]))

                    # Keep only last 100 episodes
                    if len(self.episode_max_x_pos) > self.max_window_size:
                        self.episode_max_x_pos.pop(0)
                        self.episode_flag_get.pop(0)
                        self.episode_lengths.pop(0)

                    # Reset tracking for next episode
                    self.current_episode_max_x[i] = 0.0
                    self.current_episode_length[i] = 0

        # Log statistics every check_freq steps
        if self.n_calls % self.check_freq == 0 and len(self.episode_max_x_pos) > 0:
            try:
                # Compute statistics from rolling window
                max_x_pos_mean = np.mean(self.episode_max_x_pos)
                flag_get_rate = np.mean(self.episode_flag_get)
                episode_length_mean = np.mean(self.episode_lengths)

                # Log to TensorBoard
                self.logger.record("progress/max_x_pos_mean", max_x_pos_mean)
                self.logger.record("progress/flag_get_rate", flag_get_rate)
                self.logger.record("progress/episode_length", episode_length_mean)
                self.logger.record(
                    "progress/episodes_tracked", len(self.episode_max_x_pos)
                )

                if self.verbose > 0:
                    print(f"\nProgress Stats (Step {self.num_timesteps:,}):")
                    print(f"  Avg Max X Position: {max_x_pos_mean:.1f}")
                    print(f"  Flag Get Rate: {flag_get_rate * 100:.1f}%")
                    print(f"  Avg Episode Length: {episode_length_mean:.1f}")

            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: ProgressTrackingCallback logging failed: {e}")

        return True
