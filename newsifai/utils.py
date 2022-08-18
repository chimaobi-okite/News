import json
import random
import numpy as np
from typing import Dict

def save_dict(d : Dict, filepath) -> None:
    """Save dict to a json file."""
    with open(filepath, "w") as fp:
        json.dump(d, indent=2, sort_keys=False, fp=fp)

def load_dict(filepath) -> Dict:
    """Load a dict from a json file."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d

def set_seed(seed : int = 42 ) -> None:
  '''Set seed for reproduciability'''
  np.random.seed(42)
  random.seed(42)
