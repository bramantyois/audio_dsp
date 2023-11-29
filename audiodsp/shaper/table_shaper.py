import numpy as np


class TableShaper:
    def __init__(self, data, normalize=True):
        """
        Shaper based on lookup table
        :param data: dict with keys 'input' and 'output' containing the lookup table data
        """
        if not isinstance(data, dict):
            raise ValueError("data must be a dict")

        self.x = np.array(data["input"]).ravel()
        self.y = np.array(data["output"]).ravel()
        if normalize:
            self.y /= self.y.max()

    def __call__(self, x):
        return np.interp(x, self.x, self.y)
