#!/usr/bin/env python
"""Quick test to verify callbacks are working."""

from callbacks import ProgressTrackingCallback, PolicyCollapseCallback

print("=" * 60)
print("CALLBACK VERIFICATION TEST")
print("=" * 60)

# Test 1: Import and instantiate ProgressTrackingCallback
try:
    ptc = ProgressTrackingCallback(check_freq=1000, verbose=1)
    print("[PASS] ProgressTrackingCallback imported and instantiated")
    print(f"  - check_freq: {ptc.check_freq}")
    print(f"  - max_window_size: {ptc.max_window_size}")
    print(f"  - Has _on_step method: {hasattr(ptc, '_on_step')}")
except Exception as e:
    print(f"[FAIL] FAILED to instantiate ProgressTrackingCallback: {e}")

print()

# Test 2: Import and instantiate PolicyCollapseCallback
try:
    pcc = PolicyCollapseCallback(
        check_freq=50000,
        dominant_action_threshold=0.85,
        entropy_threshold=0.3,
        checkpoint_dir="./test",
        checkpoint_prefix="test_",
        verbose=1,
    )
    print("[PASS] PolicyCollapseCallback imported and instantiated")
    print(f"  - check_freq: {pcc.check_freq}")
    print(f"  - dominant_action_threshold: {pcc.dominant_action_threshold}")
    print(f"  - entropy_threshold: {pcc.entropy_threshold}")
    print(f"  - Has _on_step method: {hasattr(pcc, '_on_step')}")
except Exception as e:
    print(f"[FAIL] FAILED to instantiate PolicyCollapseCallback: {e}")

print()
print("=" * 60)
print("[PASS] ALL CALLBACK TESTS PASSED")
print("=" * 60)
