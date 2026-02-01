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
        if 'Gym has been unmaintained' not in msg and 'np.bool8' not in msg:
            self.stream.write(msg)
    
    def flush(self):
        self.stream.flush()


sys.stderr = _SuppressGymWarning(sys.stderr)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
