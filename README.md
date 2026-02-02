# Super Mario Bros RL Agent (PPO)

<div align="center">

**[English](README.md) | [简体中文](README.zh.md)**

![banner](assets/banner.png)

### Level 1-1 Completion Demo

![Mario 1-1 Demo](assets/demo-1-1.gif)

*Trained with PPO after ~12.5M timesteps*

</div>

---

A reinforcement learning agent that learns to play Super Mario Bros using **Proximal Policy Optimization (PPO)**. This is a reimplementation of my previous DQN-based Mario agent, now using PPO for improved stability and sample efficiency.

## Version Compatibility

> [!WARNING]
> **Breaking Change in v2.0.0**
>
> | Version | Action Space | Actions | Compatible Models |
> |---------|--------------|---------|-------------------|
> | v1.x | SIMPLE_MOVEMENT | 7 | v1.x only |
> | v2.x | COMPLEX_MOVEMENT | 12 | v2.x only |
>
> **v1.x and v2.x models are NOT compatible** due to different action space sizes.
>
> v2.0 adds `down` (enter pipes), `up` (climb vines), and full left movement actions.

## Trained Models

| Level | Version | Model | Training Steps | Result |
|-------|---------|-------|---------------|--------|
| 1-1 | v1.0.0 | [Download](https://github.com/SugarmanZhu/Mario/releases/download/v1.0.0/1-1-v0.zip) | ~12.5M | Completes level |
| 1-2 | v1.1.0 | [Download](https://github.com/SugarmanZhu/Mario/releases/download/v1.1.0/1-2-v0.zip) | ~9.7M | Reaches warp zone |

> [!NOTE]
> v1.x models require [v1.1.0](https://github.com/SugarmanZhu/Mario/releases/tag/v1.1.0) or earlier code.
> See [Releases](https://github.com/SugarmanZhu/Mario/releases) for all pre-trained models.

## Action Space

<details>
<summary><b>v2.x - COMPLEX_MOVEMENT (12 actions)</b> - Current</summary>

| # | Action | Description |
|---|--------|-------------|
| 0 | `NOOP` | Do nothing |
| 1 | `right` | Walk right |
| 2 | `right + A` | Jump right |
| 3 | `right + B` | Run right |
| 4 | `right + A + B` | Sprint jump right |
| 5 | `A` | Jump in place |
| 6 | `left` | Walk left |
| 7 | `left + A` | Jump left |
| 8 | `left + B` | Run left |
| 9 | `left + A + B` | Sprint jump left |
| 10 | `down` | Crouch / Enter pipe |
| 11 | `up` | Climb vine / Enter door |

</details>

<details>
<summary><b>v1.x - SIMPLE_MOVEMENT (7 actions)</b> - Legacy</summary>

| # | Action | Description |
|---|--------|-------------|
| 0 | `NOOP` | Do nothing |
| 1 | `right` | Walk right |
| 2 | `right + A` | Jump right |
| 3 | `right + B` | Run right |
| 4 | `right + A + B` | Sprint jump right |
| 5 | `A` | Jump in place |
| 6 | `left` | Walk left |

</details>

## Why PPO over DQN?

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Stability** | Can be unstable, sensitive to hyperparameters | More stable training with clipped objectives |
| **Sample Efficiency** | Requires large replay buffer | On-policy, no replay buffer needed |
| **Parallelization** | Single environment | Easily parallelizes across multiple environments |
| **Convergence** | Can oscillate or diverge | Smoother, more reliable convergence |

## Features

- **Multi-Level Training**: Train on multiple levels simultaneously to prevent catastrophic forgetting
- **Parallel Training**: Uses `SubprocVecEnv` for true multi-process parallelism
- **Automatic Checkpointing**: Saves model every 100K steps with resume support
- **Best Model Tracking**: Automatically saves the best performing model
- **TensorBoard Integration**: Real-time training visualization
- **Video Recording**: Record gameplay as GIF/MP4/WebP
- **Policy Collapse Detection**: Automatic detection and recovery

## Hardware Used

- **GPU**: NVIDIA GeForce RTX 5090 (32GB VRAM)
- **RAM**: 64GB
- **OS**: Windows 11

## Quick Start

```bash
# Clone the repository
git clone https://github.com/SugarmanZhu/Mario.git
cd Mario

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA (RTX 40/50 series)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Test environment
python test_env.py

# Train agent (--env supports shorthand: '1-1' = 'SuperMarioBros-1-1-v0')
python train_ppo.py --timesteps 10000000 --n-envs 16

# Watch trained agent play
python train_ppo.py --mode play --model ./mario_models/1-1/best/best_model.zip --slow

# Record gameplay video
python record_video.py --model ./mario_models/1-1/best/best_model.zip --output demo.gif
```

## Recording Gameplay

```bash
# Record as GIF (default)
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.gif

# Record as MP4
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.mp4 --fps 30

# Record as WebP (smaller file size)
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.webp --quality 90

# Record multiple episodes, keep best
python record_video.py --model mario_models/flag/1-1-v0.zip --output demo.gif --episodes 5 --best
```

## Project Structure

```
Mario/
├── train_ppo.py        # Main training and inference script
├── record_video.py     # Record gameplay as GIF/MP4/WebP
├── wrappers.py         # Custom environment wrappers
├── callbacks.py        # Training callbacks (collapse detection)
├── utils.py            # Utility functions
├── diagnose_policy.py  # Tool to analyze policy health
├── requirements.txt    # Python dependencies
├── mario_models/       # Saved model checkpoints
│   ├── best/           # Best model based on evaluation
│   └── flag/           # Models that complete levels
└── mario_logs/         # TensorBoard logs
```

## Training

Start a new training run:
```bash
python train_ppo.py --timesteps 10000000 --n-envs 16
```

Resume from a checkpoint:
```bash
python train_ppo.py --resume ./mario_models/1-1/checkpoints/.../checkpoint.zip --timesteps 20000000
```

### Multi-Level Training

Train on multiple levels simultaneously to prevent catastrophic forgetting:
```bash
# Train on 1-1 and 1-2 together (shorthand supported)
python train_ppo.py --env "1-1,1-2" --timesteps 15000000

# Resume multi-level training with higher entropy for exploration
python train_ppo.py --resume ./mario_models/1-2/checkpoints/.../checkpoint.zip --env "1-1,1-2" --timesteps 15000000 --ent-coef 0.07
```

Multi-level training:
- Distributes parallel workers across levels (round-robin)
- Evaluates on all levels to detect forgetting
- Saves to `mario_models/multi-1-1-1-2/` folder

### Play Mode

Watch a trained agent play:
```bash
# Single level
python train_ppo.py --mode play --model ./mario_models/1-1/best/best_model.zip --slow

# Multiple levels (plays each level once, sequentially)
python train_ppo.py --mode play --model ./mario_models/multi-1-1-1-2/best/best_model.zip --env "1-1,1-2" --slow
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `train` | `train` or `play` |
| `--env` | `1-1` | Environment ID(s), supports shorthand (e.g., `1-1`, `1-1,1-2`) |
| `--timesteps` | `2000000` | Total training timesteps |
| `--n-envs` | `16` | Number of parallel environments |
| `--model` | `None` | Model path for play mode |
| `--resume` | `None` | Checkpoint path to resume training |
| `--lr` | `0.0001` | Learning rate |
| `--ent-coef` | `0.05` | Entropy coefficient (higher = more exploration) |
| `--slow` | `False` | Slow down playback for viewing |

## PPO Hyperparameters

```python
n_steps = 2048          # Steps per environment per update
batch_size = 512        # Minibatch size
n_epochs = 10           # Epochs per update
gamma = 0.99            # Discount factor
gae_lambda = 0.95       # GAE lambda
clip_range = 0.2        # PPO clipping parameter
ent_coef = 0.05         # Entropy coefficient
vf_coef = 0.5           # Value function coefficient
max_grad_norm = 0.5     # Gradient clipping
learning_rate = 1e-4    # With linear decay
```

## Custom Reward Shaping

| Reward | Value | Description |
|--------|-------|-------------|
| Forward progress | `+0.1 × Δx` | Reward for moving right |
| Stuck penalty | `-0.5` | After 10+ frames without moving |
| Time penalty | `-0.01` | Per step (encourages speed) |
| Flag bonus | `+100` | Reaching the flagpole |
| Death penalty | `-50` | Dying without reaching flag |
| Score reward | `+0.01 × Δscore` | Killing enemies, power-ups |
| Coin bonus | `+5.0` per coin | Collecting coins |

## Training Progress

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
1. Run `python diagnose_policy.py --model YOUR_MODEL`
2. Find healthy checkpoint: `python diagnose_policy.py --find-healthy`
3. Resume from it: `python train_ppo.py --resume HEALTHY_CHECKPOINT`

### Training Freezes
The `PolicyCollapseCallback` was fixed to avoid deadlocks with `SubprocVecEnv`. If you experience freezes, ensure you have the latest version.

## License

MIT

## Acknowledgments

- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros) - Mario environment
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) - PPO implementation
- [nes-py](https://github.com/Kautenja/nes-py) - NES emulator
