# Super Mario Bros RL Agent (PPO)

A reinforcement learning agent that learns to play Super Mario Bros using **Proximal Policy Optimization (PPO)**. This is a reimplementation of my previous DQN-based Mario agent, now using PPO for improved stability and sample efficiency.

<img width="1536" height="274" alt="banner" src="https://github.com/user-attachments/assets/aad95dcd-9c26-41b3-8287-d334a0bab3ec" />

## Why PPO over DQN?

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Stability** | Can be unstable, sensitive to hyperparameters | More stable training with clipped objectives |
| **Sample Efficiency** | Requires large replay buffer | On-policy, no replay buffer needed |
| **Parallelization** | Single environment | Easily parallelizes across multiple environments |
| **Convergence** | Can oscillate or diverge | Smoother, more reliable convergence |

## Features

- **Parallel Training**: Uses `SubprocVecEnv` for true multi-process parallelism (tested with 48 environments on RTX 5090)
- **Automatic Checkpointing**: Saves model every 100K steps with resume support
- **Best Model Tracking**: Automatically saves the best performing model based on evaluation
- **TensorBoard Integration**: Real-time training visualization
- **Custom Preprocessing Pipeline**: Optimized observation wrappers for efficient learning
- **Custom Reward Shaping**: Prevents policy collapse and encourages diverse behavior

## Hardware Used

- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **RAM**: 64GB
- **OS**: Windows 11

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### 2. Install PyTorch with CUDA support

For RTX 40/50 series (CUDA 12.x):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

For older GPUs (CUDA 11.8):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note**: NumPy must be <2.0 for compatibility with the old Gym API used by `gym-super-mario-bros`.

## Project Structure

```
Mario/
├── train_ppo.py        # Main training and inference script
├── wrappers.py         # Custom environment wrappers
├── diagnose_policy.py  # Tool to analyze policy health and detect collapse
├── requirements.txt    # Python dependencies
├── mario_models/       # Saved model checkpoints
│   └── best/           # Best model based on evaluation
└── mario_logs/         # TensorBoard logs
```

## Usage

### Training

Start a new training run:
```bash
python train_ppo.py --mode train --timesteps 10000000 --n-envs 16
```

Resume from a checkpoint:
```bash
python train_ppo.py --mode train --resume ./mario_models/best/best_model.zip --timesteps 2000000
```

### Playing/Testing

Watch the trained agent play:
```bash
python train_ppo.py --mode play --model ./mario_models/best/best_model
```

Watch in slow motion (easier for humans to follow):
```bash
python train_ppo.py --mode play --model ./mario_models/best/best_model --slow
```

### Diagnosing Policy Health

Check if a model has policy collapse:
```bash
python diagnose_policy.py --model ./mario_models/best/best_model.zip
```

Find the last healthy checkpoint:
```bash
python diagnose_policy.py --find-healthy
```

### Monitoring Training

Launch TensorBoard to visualize training progress:
```bash
tensorboard --logdir ./mario_logs
```

Then open http://localhost:6006 in your browser.

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` or `play` |
| `--env` | `SuperMarioBros-1-1-v0` | Environment ID |
| `--timesteps` | `2000000` | Total training timesteps |
| `--n-envs` | `16` | Number of parallel environments |
| `--model` | `None` | Model path for play mode |
| `--resume` | `None` | Checkpoint path to resume training |
| `--lr` | `0.0001` | Learning rate |
| `--slow` | `False` | Slow down playback for human viewing |

## Custom Reward Shaping

The `CustomRewardWrapper` adds reward shaping to prevent policy collapse and encourage good behavior:

| Reward | Value | Description |
|--------|-------|-------------|
| Forward progress | `+0.1 × Δx` | Reward for moving right |
| Stuck penalty | `-0.5` | After 10+ frames without moving |
| Time penalty | `-0.01` | Per step (encourages speed) |
| Flag bonus | `+100` | Reaching the flagpole |
| Death penalty | `-50` | Dying without reaching flag |
| Score reward | `+0.01 × Δscore` | Killing enemies, power-ups |
| Coin bonus | `+5.0` per coin | Collecting coins |

## Preprocessing Pipeline

The observation goes through several transformations before being fed to the neural network:

1. **Frame Skip (4)**: Repeat each action for 4 frames, sum rewards
2. **Grayscale**: Convert RGB to single-channel grayscale
3. **Resize (84x84)**: Downscale to 84x84 pixels
4. **Normalize**: Scale pixel values from [0, 255] to [0, 1]
5. **Frame Stack (4)**: Stack 4 consecutive frames for temporal information
6. **Transpose**: Convert from (H, W, C) to (C, H, W) for PyTorch

**Final observation shape**: `(4, 84, 84)` - 4 stacked 84x84 grayscale frames

## PPO Hyperparameters

```python
n_steps = 2048          # Steps per environment per update
batch_size = 512        # Minibatch size (optimized for RTX 5090)
n_epochs = 10           # Epochs per update
gamma = 0.99            # Discount factor
gae_lambda = 0.95       # GAE lambda for advantage estimation
clip_range = 0.2        # PPO clipping parameter
ent_coef = 0.05         # Entropy coefficient (prevents policy collapse)
vf_coef = 0.5           # Value function coefficient
max_grad_norm = 0.5     # Gradient clipping
learning_rate = 1e-4    # With linear decay schedule
```

## Action Space

Using `SIMPLE_MOVEMENT` (7 discrete actions):

| Action | Description |
|--------|-------------|
| 0 | NOOP |
| 1 | Right |
| 2 | Right + A (jump) |
| 3 | Right + B (run) |
| 4 | Right + A + B (run + jump) |
| 5 | A (jump) |
| 6 | Left |

## Training Progress

Typical learning milestones (may vary):

| Steps | Expected Behavior |
|-------|-------------------|
| 0-500K | Random exploration, learns to move right |
| 500K-2M | Starts avoiding basic obstacles |
| 2M-5M | Learns to jump over pipes and gaps |
| 5M-10M | Can complete level 1-1 occasionally |
| 10M+ | Consistent level completion |

## Troubleshooting

### Policy Collapse

If Mario stops moving or always takes the same action:

1. **Diagnose**: Run `python diagnose_policy.py --model YOUR_MODEL`
2. **Find healthy checkpoint**: Run `python diagnose_policy.py --find-healthy`
3. **Resume from healthy checkpoint**: `python train_ppo.py --mode train --resume HEALTHY_CHECKPOINT`

Signs of policy collapse:
- Dominant action > 80%
- Low action entropy (< 0.5)
- Mario not making progress (low x_pos)

### Automatic Collapse Detection & Recovery

Training now includes a **PolicyCollapseCallback** that automatically monitors for policy collapse and rolls back to a healthy checkpoint when detected.

**Detection triggers:**
- Dominant action > 85% of all actions
- Entropy drops below 0.3

**How it works:**
1. Every 50K steps, the callback samples the policy's action distribution
2. If collapse is detected, it automatically loads the last healthy checkpoint
3. Training continues from the healthy checkpoint with full context preserved

**TensorBoard metrics:**
- `policy_health/dominant_action` - Most frequently chosen action
- `policy_health/dominant_ratio` - % of times the dominant action is chosen  
- `policy_health/entropy` - Action distribution entropy
- `policy_health/collapse_count` - Number of collapses detected
- `policy_health/recovery_count` - Number of successful recoveries

### High RAM Usage

Each parallel environment runs a full NES emulator. Reduce `--n-envs` if you run out of memory.

## Resource Usage

With 48 parallel environments on RTX 5090:

| Resource | Usage |
|----------|-------|
| CPU | 10-40% (fluctuates between rollout/update phases) |
| RAM | ~50GB |
| GPU | ~30% |
| VRAM | ~3GB |

## Technical Notes

### Gym Compatibility

This project uses `gym==0.26.2` with `apply_api_compatibility=True` because:
- `gym-super-mario-bros` is built on the old Gym API
- NumPy 2.x breaks compatibility, so we pin `numpy<2.0`
- The compatibility flag bridges the old and new API

### Known Issues

- **High RAM usage**: Each parallel environment runs a full NES emulator
- **CPU fluctuation**: Normal behavior - PPO alternates between rollout collection (high CPU) and policy updates (high GPU)

## License

MIT

## Acknowledgments

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - Mario environment
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO implementation
- [nes-py](https://github.com/Kautenja/nes-py) - NES emulator
