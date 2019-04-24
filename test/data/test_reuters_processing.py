import unittest
from data.reuters_processing import ReutersData


class TestReutersData(unittest.TestCase):
    def setUp(self):
        self.data = ReutersData('../../data/' + ReutersData.default_folder)
        self.data.transform()

    def test_init(self):
        # verify size after init processing
        self.assertTrue(len(self.data._x_text) == 19716, '_x size is not equal')
        self.assertTrue(self.data._y.shape == (19716,), '_y size is not equal')