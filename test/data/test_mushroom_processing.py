import unittest
import numpy as np
from data.mushroom_processing import MushroomData


class TestMushroomData(unittest.TestCase):
    # p, x, s, n, t, p, f, c, n, k, e, e, s, s, w, w, p, w, o, p, k, s, u
    # e, x, s, y, t, a, f, c, b, k, e, c, s, s, w, w, p, w, o, p, n, n, g
    # e, b, s, w, t, l, f, c, b, n, e, c, s, s, w, w, p, w, o, p, n, n, m
    # p, x, y, w, t, p, f, c, n, n, e, e, s, s, w, w, p, w, o, p, k, s, u
    # e, x, s, g, f, n, f, w, b, k, t, e, s, s, w, w, p, w, o, e, n, a, g
    # e, x, y, y, t, a, f, c, b, n, e, c, s, s, w, w, p, w, o, p, k, n, g
    # e, b, s, w, t, a, f, c, b, g, e, c, s, s, w, w, p, w, o, p, k, n, m
    # e, b, y, w, t, l, f, c, b, n, e, c, s, s, w, w, p, w, o, p, n, s, m
    # p, x, y, w, t, p, f, c, n, p, e, e, s, s, w, w, p, w, o, p, k, v, g
    # e, b, s, y, t, a, f, c, b, g, e, c, s, s, w, w, p, w, o, p, k, s, m

    def setUp(self):
        self.data = MushroomData('../../data/mushroom/agaricus-lepiota.data.preprocessing.test.txt')
        self.data.transform()

    def test_transform(self):
        # size verify
        self.assertTrue(np.shape(self.data._x) == (10, 20),
                        'Wrong train_test split size')
        # verify label
        expected_y = [0,1,1,0,1,1,1,1,0,1]
        self.assertTrue(all(self.data._y == expected_y))

    def test_extract(self):
        # also verify labels do not in dataset
        x_train, _, x_test, _ = self.data.extract(np.arange(7), np.arange(7,10))
        expected_x_train = np.array([
            [3/5, 4/5, 0,   4/6, 0, 5/7, 4/6, 0, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 0, 0],
            [3/5, 4/5, 1,   4/6, 1, 5/7, 4/6, 1, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 1,   1, 1],
            [1,   4/5, 2/3, 4/6, 1, 5/7, 4/6, 1, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 1,   1, 1],
            [3/5, 1/2, 2/3, 4/6, 0, 5/7, 4/6, 0, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 0, 0],
            [3/5, 4/5, 1,   1,   1, 5/7, 1,   1, 2/3, 1,   5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 1,   1,   1, 1],
            [3/5, 1/2, 1,   4/6, 1, 5/7, 4/6, 1, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 1, 1],
            [1,   4/5, 2/3, 4/6, 1, 5/7, 4/6, 1, 1,   4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 1, 1],
        ])
        expected_x_test = np.array([
            [1,   1/2, 2/3, 4/6, 1, 5/7, 4/6, 1, 2/3, 4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 1,   0, 1],
            [3/5, 1/2, 2/3, 4/6, 0, 5/7, 4/6, 0, 0,   4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 0, 1],
            [1,   4/5, 1,   4/6, 1, 5/7, 4/6, 1, 1,   4/6, 5/7, 5/7, 5/7, 5/7, 5/7, 5/7, 4/6, 2/4, 0, 1]
        ])
        self.assertTrue(np.array_equal(expected_x_train, x_train))
        self.assertTrue(np.array_equal(expected_x_test, x_test))
        self.assertTrue(np.array_equal(expected_x_test, x_test))
