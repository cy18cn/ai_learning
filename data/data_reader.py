import numpy as np
from pathlib import Path


def load_data(file_name):
    file = Path(file_name)
    if file.exists():
        data = np.load(file)
        x = data["x"]
        y = data["label"]

    return x, y
