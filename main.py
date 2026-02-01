"""
Mario RL Agent - Main Entry Point

This project trains a reinforcement learning agent to play Super Mario Bros
using PPO (Proximal Policy Optimization) from Stable-Baselines3.

Quick Start:
1. Install dependencies: pip install -r requirements.txt
2. Test environment:     python test_env.py
3. Train agent:          python train_ppo.py --mode train
4. Play with agent:      python train_ppo.py --mode play --model ./mario_models/best/best_model
"""

import sys


def main():
    """Main entry point with simple menu."""
    print("=" * 60)
    print("  Super Mario Bros RL Agent")
    print("=" * 60)
    print("\nAvailable commands:")
    print("  1. python test_env.py       - Test environment setup")
    print("  2. python train_ppo.py      - Train PPO agent")
    print("  3. python train_ppo.py --mode play --model PATH")
    print("\nSetup:")
    print("  pip install -r requirements.txt")
    print("\nEnvironment Info:")
    
    try:
        import gym
        print(f"  - gym version: {gym.__version__}")
    except ImportError:
        print("  - gym: NOT INSTALLED")
    
    try:
        import gym_super_mario_bros
        print(f"  - gym-super-mario-bros: INSTALLED")
    except ImportError:
        print("  - gym-super-mario-bros: NOT INSTALLED")
    
    try:
        import stable_baselines3
        print(f"  - stable-baselines3: {stable_baselines3.__version__}")
    except ImportError:
        print("  - stable-baselines3: NOT INSTALLED")
    
    try:
        import torch
        print(f"  - PyTorch: {torch.__version__}")
        print(f"  - CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  - CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("  - PyTorch: NOT INSTALLED")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
