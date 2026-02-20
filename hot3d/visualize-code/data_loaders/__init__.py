import os


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_HOT3D_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_REAL_DATA_LOADERS_DIR = os.path.join(_HOT3D_ROOT, "data_loaders")

# Make `from data_loaders.xxx import ...` resolve to the real package even when
# scripts are launched from `visualize-code` with top-level imports.
__path__ = [p for p in (__path__ if "__path__" in globals() else [])]
if _REAL_DATA_LOADERS_DIR not in __path__:
    __path__.append(_REAL_DATA_LOADERS_DIR)
