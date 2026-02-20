import importlib.util
import os


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_HOT3D_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_MANO_PATH = os.path.join(_HOT3D_ROOT, "mano.py")

_spec = importlib.util.spec_from_file_location("_hot3d_mano", _MANO_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Failed to load mano module from {_MANO_PATH}")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

__all__ = [name for name in globals() if not name.startswith("_")]
