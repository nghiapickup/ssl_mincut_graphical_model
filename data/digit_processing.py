"""
    @nghia nh
    ---
    Preprocessing Digit data
"""

import logging

import numpy as np
from sklearn.datasets import load_digits
from data.base_processing import BaseDataProcessor


class DigitData(BaseDataProcessor):
    """
    Binary classification task: Odd vs Even

    Classes             10
    Samples per class   ~180
    Samples total       1797
    Dimensionality      64
    Features            integers 0-16
    """

    def __init__(self):
        logging.info('DigitData __init__')

        BaseDataProcessor.__init__(self)
        self._x, self._y = load_digits(return_X_y=True)
        self._x_number = len(self._x)

    def transform(self):
        logging.info('DigitData transform binary task: Odd vs Even')

        self._x = np.array(self._x)
        self._y = np.fromiter((x%2 for x in self._y), dtype=int)

        return self
