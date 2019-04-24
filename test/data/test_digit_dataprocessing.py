import unittest
from data.digit_processing import DigitData


class TestDigitData(unittest.TestCase):
    def setUp(self):
        self.data = DigitData()
        self.data.transform()

    def test_init(self):
        # verify size after init processing
        self.assertTrue(self.data._x.shape == (1797, 64), '_x size is not equal')
        self.assertTrue(self.data._y.shape == (1797,), '_y size is not equal')
        self.assertTrue(all(x in [0,1] for x in self.data._y))