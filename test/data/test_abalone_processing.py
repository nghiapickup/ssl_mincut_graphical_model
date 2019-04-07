import unittest
import numpy as np
from data.abalone_processing import AbaloneData


class TestAbaloneData(unittest.TestCase):
    # M, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15
    # M, 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07, 7
    # F, 0.53, 0.42, 0.135, 0.677, 0.2565, 0.1415, 0.21, 1
    # M, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155, 10
    # I, 0.33, 0.255, 0.08, 0.205, 0.0895, 0.0395, 0.055, 7
    # I, 0.425, 0.3, 0.095, 0.3515, 0.141, 0.0775, 0.12, 8
    # F, 0.53, 0.415, 0.15, 0.7775, 0.237, 0.1415, 0.33, 20
    # F, 0.545, 0.425, 0.125, 0.768, 0.294, 0.1495, 0.26, 16
    # M, 0.475, 0.37, 0.125, 0.5095, 0.2165, 0.1125, 0.165, 9
    # F, 0.55, 0.44, 0.15, 0.8945, 0.3145, 0.151, 0.32, 19
    #
    # after init
    #
    # 1, 0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 15
    # 1, 0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07, 7
    # 1, 0.44, 0.365, 0.125, 0.516, 0.2155, 0.114, 0.155, 10
    # 0, 0.33, 0.255, 0.08, 0.205, 0.0895, 0.0395, 0.055, 7
    # 0, 0.425, 0.3, 0.095, 0.3515, 0.141, 0.0775, 0.12, 8
    # 1, 0.475, 0.37, 0.125, 0.5095, 0.2165, 0.1125, 0.165, 9

    def setUp(self):
        self.data = AbaloneData('../../data/abalone/abalone.data.preprocessing.test.txt')

    def test_init(self):
        # verify size after init processing
        self.assertTrue(self.data._abalone_df.shape == (6,9), 'Data size is not equal')
        # verify Sex feature
        expected_sex = np.array([1, 1, 1, 0, 0, 1])
        self.assertTrue(np.array_equal(expected_sex, self.data._abalone_df['Sex']), 'Wrong Sex feature scalarizing')
        # verify Rings feature
        expected_rings = np.array([15, 7, 10, 7, 8, 9])
        self.assertTrue(np.array_equal(expected_rings, self.data._abalone_df['Rings']), 'Wrong Rings feature')

    def test_transform(self):
        self.data.positive_labels = [7]
        self.data.transform()
        # size verify
        self.assertTrue((len(self.data._x), len(self.data._y)) == (6, 6),
                        'Wrong train_test split size')
        # verify label
        expected_y = [0,1,0,1,0,0]
        self.assertTrue(np.array_equal(self.data._y, expected_y))

        # also verify labels do not in dataset
        self.data.positive_labels = [7, 20, 8]
        self.data.transform()
        # size verify
        self.assertTrue((len(self.data._x), len(self.data._y)) == (6, 6),
                        'Wrong train_test split size')
        # verify label
        expected_y = np.array([0,1,0,1,1,0])
        self.assertTrue(np.array_equal(self.data._y,expected_y))
