"""
Suppress noisy gym/numpy deprecation warnings.
Import this module before any gym imports.
"""
import warnings
import sys


class _SuppressGymWarning:
    def __init__(self, stream):
        self.stream = stream
    
    def write(self, msg):
        # Skip empty or whitespace-only messages
        if not msg or msg.isspace():
            return
        # Skip known warning patterns
        suppress = [
            'Gym has been unmaintained',
            'bool8',
            'DeprecationWarning',
            'UserWarning',
        ]
        if not any(s in msg for s in suppress):
            self.stream.write(msg)
    
    def flush(self):
        self.stream.flush()


sys.stderr = _SuppressGymWarning(sys.stderr)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
