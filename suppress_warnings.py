"""
Suppress noisy gym/numpy deprecation warnings.
Import this module before any gym imports.
"""
import sys
import warnings


class _SuppressGymWarning:
    def __init__(self, stream):
        self._stream = stream

    def write(self, msg):
        if not msg or msg.isspace():
            return
        suppress = [
            'Gym has been unmaintained',
            'bool8',
            'DeprecationWarning',
            'UserWarning',
        ]
        if not any(s in msg for s in suppress):
            self._stream.write(msg)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


sys.stderr = _SuppressGymWarning(sys.stderr)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
