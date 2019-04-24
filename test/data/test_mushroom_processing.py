import unittest
from io import StringIO

import numpy as np
from source import data_dir
from data.mushroom_processing import MushroomData


class TestMushroomData(unittest.TestCase):
    def setUp(self):
        tiny_mushroom = np.array([
            ['p', 'x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e',
             'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'],
            ['e', 'x', 's', 'y', 't', 'a', 'f', 'c', 'b', 'k', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'g'],
            ['e', 'b', 's', 'w', 't', 'l', 'f', 'c', 'b', 'n', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'm'],
            ['p', 'x', 'y', 'w', 't', 'p', 'f', 'c', 'n', 'n', 'e',
             'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'],
            ['e', 'x', 's', 'g', 'f', 'n', 'f', 'w', 'b', 'k', 't',
             'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'e', 'n', 'a', 'g'],
            ['e', 'x', 'y', 'y', 't', 'a', 'f', 'c', 'b', 'n', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 'n', 'g'],
            ['e', 'b', 's', 'w', 't', 'a', 'f', 'c', 'b', 'g', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 'n', 'm'],
            ['e', 'b', 'y', 'w', 't', 'l', 'f', 'c', 'b', 'n', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 's', 'm'],
            ['p', 'x', 'y', 'w', 't', 'p', 'f', 'c', 'n', 'p', 'e',
             'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 'v', 'g'],
            ['e', 'b', 's', 'y', 't', 'a', 'f', 'c', 'b', 'g', 'e',
             'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'm']
        ]).reshape((10, 23))

        # verify on small amount of data
        self.input_stream = StringIO()
        np.savetxt(self.input_stream, tiny_mushroom, fmt='%s', delimiter=',')
        self.input_stream.seek(0)
        self.mushroom = MushroomData(self.input_stream)

    def tearDown(self):
        self.input_stream.close()

    def test_input(self):
        # verify basic info
        self.mushroom = MushroomData('../../' + data_dir['mushroom'])
        # data after remove 2 unused features
        self.assertEqual(self.mushroom._mushroom_df.shape, (8124, 20+1))

    def test_transform(self):
        self.mushroom.transform()
        # size verify
        self.assertEqual(self.mushroom._x.shape, (10, 20),
                        'Wrong train_test split size')
        # verify label
        expected_y = np.array([0,1,1,0,1,1,1,1,0,1])
        self.assertTrue(np.array_equal(self.mushroom._y, expected_y))

    def test_extract(self):
        self.mushroom.transform()

        # also verify labels do not in dataset
        x_train, _, x_test, _ = self.mushroom.extract(np.arange(7), np.arange(7,10))
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
