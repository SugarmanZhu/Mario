"""
Test script to verify the Super Mario Bros environment works correctly.
This tests the compatibility layer between old gym and new gymnasium API.
"""
# Suppress ALL gym warnings before import
import warnings
import sys

# Suppress the gym deprecation message (it prints to stderr)
class SuppressGymWarning:
    def __init__(self, stream):
        self.stream = stream
    def write(self, msg):
        if 'Gym has been unmaintained' not in msg and 'np.bool8' not in msg:
            self.stream.write(msg)
    def flush(self):
        self.stream.flush()

sys.stderr = SuppressGymWarning(sys.stderr)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

import time


def test_basic_environment():
    """Test if the environment can be created and run basic steps."""
    print("=" * 60)
    print("Testing Super Mario Bros Environment Setup")
    print("=" * 60)
    
    # Step 1: Import required packages
    print("\n[1] Importing packages...")
    try:
        import gym
        print(f"    gym version: {gym.__version__}")
    except ImportError as e:
        print(f"    ERROR: Failed to import gym: {e}")
        print("    Run: pip install gym==0.26.2")
        return False
    
    try:
        import gym_super_mario_bros
        from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
        print(f"    gym_super_mario_bros imported successfully")
    except ImportError as e:
        print(f"    ERROR: Failed to import gym_super_mario_bros: {e}")
        print("    Run: pip install gym-super-mario-bros==7.4.0")
        return False
    
    try:
        from nes_py.wrappers import JoypadSpace
        print(f"    nes_py imported successfully")
    except ImportError as e:
        print(f"    ERROR: Failed to import nes_py: {e}")
        print("    Run: pip install nes-py")
        return False
    
    # Step 2: Create environment with compatibility mode
    print("\n[2] Creating environment with apply_api_compatibility=True...")
    try:
        env = gym.make(
            'SuperMarioBros-v0',
            apply_api_compatibility=True,
            render_mode='rgb_array'
        )
        print(f"    Environment created: {env.spec.id}")
        print(f"    Observation space: {env.observation_space}")
        print(f"    Action space (raw): {env.action_space}")
    except Exception as e:
        print(f"    ERROR creating environment: {e}")
        return False
    
    # Step 3: Apply JoypadSpace wrapper to reduce action space
    print("\n[3] Applying JoypadSpace wrapper...")
    try:
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        print(f"    JoypadSpace applied")
        print(f"    Action space (simplified): {env.action_space}")
        print(f"    Actions: {SIMPLE_MOVEMENT}")
    except Exception as e:
        print(f"    ERROR applying JoypadSpace: {e}")
        return False
    
    # Step 4: Test reset
    print("\n[4] Testing env.reset()...")
    try:
        obs, info = env.reset()
        print(f"    Reset returned (obs, info) - New API")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Info keys: {list(info.keys())}")
    except Exception as e:
        print(f"    ERROR during reset: {e}")
        return False
    
    # Step 5: Test step
    print("\n[5] Testing env.step()...")
    try:
        # Take a random action
        action = env.action_space.sample()
        result = env.step(action)
        
        # Check return format
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            print(f"    Step returned 5 values - New API (with compatibility)")
        else:
            print(f"    Unexpected step return length: {len(result)}")
            return False
            
        print(f"    Action taken: {action}")
        print(f"    Observation shape: {obs.shape}")
        print(f"    Reward: {reward}")
        print(f"    Done: {done}")
        print(f"    Info: {info}")
    except Exception as e:
        print(f"    ERROR during step: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 6: Run a short episode
    print("\n[6] Running 100 random steps...")
    try:
        total_reward = 0
        steps = 0
        start_time = time.time()
        
        for i in range(100):
            action = env.action_space.sample()
            result = env.step(action)
            
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
            
            total_reward += reward
            steps += 1
            
            if done:
                env.reset()
        
        elapsed = time.time() - start_time
        fps = steps / elapsed if elapsed > 0 else 0
        
        print(f"    Completed {steps} steps")
        print(f"    Total reward: {total_reward:.2f}")
        print(f"    FPS: {fps:.1f}")
        print(f"    Mario x_pos: {info.get('x_pos', 'N/A')}")
    except Exception as e:
        print(f"    ERROR during episode: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Cleanup
    env.close()
    
    print("\n" + "=" * 60)
    print("SUCCESS! Environment is working correctly.")
    print("=" * 60)
    return True


def test_with_rendering():
    """Test with human rendering (opens a window)."""
    print("\nTesting with visual rendering...")
    print("(Close the window or press Ctrl+C to stop)\n")
    
    import gym
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    from nes_py.wrappers import JoypadSpace
    
    try:
        env = gym.make(
            'SuperMarioBros-v0',
            apply_api_compatibility=True,
            render_mode='human'
        )
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        
        env.reset()
        
        for _ in range(500):
            action = env.action_space.sample()
            result = env.step(action)
            
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated

            if done:
                env.reset()
        
        env.close()
        print("Visual test completed!")
        
    except Exception as e:
        print(f"Visual test error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    success = test_basic_environment()
    
    if success:
        print("\n" + "-" * 60)
        user_input = input("Run visual test? (y/n): ").strip().lower()
        if user_input == 'y':
            test_with_rendering()
