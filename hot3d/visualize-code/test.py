import argparse
import json
import inspect
import re
from typing import Optional
if not hasattr(inspect, "getargspec"):
    # Compatibility for chumpy on Python 3.11+
    inspect.getargspec = inspect.getfullargspec  # type: ignore[attr-defined]
import numpy as np
# NumPy 1.24+ compatibility for legacy packages (e.g., chumpy)
for _name, _type in [("bool", bool), ("int", int), ("float", float), ("object", object), ("str", str)]:
    # Use __dict__ to avoid triggering NumPy deprecation warnings during attribute access
    if _name not in np.__dict__:
        setattr(np, _name, _type)

from projectaria_tools.core.sophus import SE3
from projectaria_tools.utils.rerun_helpers import ToTransform3D
import trimesh

from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

import os
import torch
import numpy as np

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from rot import *

import rerun as rr
import pickle

from hot3d_action_dataset import Hot3DActionDataset as _Hot3DActionDataset

# The saved pickle references "__main__.Hot3DActionDataset". Make sure this
# name exists so pickle can resolve the dataset class.
globals()["Hot3DActionDataset"] = _Hot3DActionDataset

rr.init("Input Data", spawn=True)

home = os.path.expanduser("~")
with open(os.path.join(home, "Desktop/hot3d_vis/a_t.pkl"), "rb") as f:
    item_list = pickle.load(f)

for item_idx, item in enumerate(item_list):
    joints = item["joints"][0]  # (T, 21, 3)
    obj_vertices = item["obj_vertices"][0].reshape(1024, -1)  # (T, V, 3)
    joints_label = f"sample_{item_idx}/joints"
    obj_label = f"sample_{item_idx}/object"

    for frame_idx, frame in enumerate(joints):
        rr.set_time_sequence("frame", frame_idx)
        rr.log(
            joints_label,
            rr.Points3D(
                frame,
                radii=0.01,
                colors=[1.0, 0.0, 0.0, 1.0],
            ),
        )


        rr.log(
            obj_label,
            rr.Points3D(
                obj_vertices,
                radii=0.005,
                colors=[0.0, 0.5, 1.0, 1.0],
            ),
        )
