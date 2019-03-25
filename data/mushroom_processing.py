"""
    @nghia nh
    Preprocessing Mushroom data
    ---

    All features are categotical data. Then we need to convert them into
    a reasonable meaning which can calc the distance. For each unique value of
    one feature, we compute the proportion between edible and poisonous class
    only on training data.
"""

import unittest
import logging

import pandas as pd
import numpy as np


class MushroomData:
    default_file_name = 'agaricus-lepiota.data.txt'
    data_header = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
              'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
              'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring',
              'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']

    def __init__(self, data_file_dir):
        logging.info('MushroomData __init__')
        self.mushroom_df = pd.read_csv(data_file_dir, names = self.data_header)
        # Map classes feature
        self.mushroom_df['class'] = self.mushroom_df['class'].map({'p':0, 'e':1})
        # Drop one value and missing features (missing rate > 1/4)
        self.mushroom_df = self.mushroom_df.drop(columns=['veil-type', 'stalk-root'])

    def get_basic_data(self):
        """
        return basic data without converting yet
        :return: ndarray pair (x,y)
        """
        logging.info('MushroomData get_basic_data')
        x = self.mushroom_df.drop(columns='class').values
        y = self.mushroom_df['class'].values
        return x, y

    def get_scalar_data(self, train_index, test_index):
        """
        Convert categorical value to scalar.
        Each unique value in one feature will transfer to its proportion of edible (e/(e + p)).
        Notice that this calculation only works on train set and apply the same scale for test set
        :param train_index: indices of instances (in full data set) using for train set
        :param test_index: indices of instances (in full data set) using for test set
        :return:
        """
        logging.info('MushroomData get_scalar_data')
        x = self.mushroom_df.iloc[train_index]
        x_train = x.drop(columns=['class'])
        x_test = self.mushroom_df.iloc[test_index].drop(columns=['class'])

        # count unique value apprance in each feature by class labels
        for feature in x_train:
            feature_map = {}
            feature_value_size = x.groupby(feature).size()
            for feature_value, group_indices in x.groupby([feature, 'class']):
                if feature_value[1] == 1:  # verify on edible value
                    feature_map[feature_value[0]] = len(group_indices) / feature_value_size[feature_value[0]]
                elif feature_value[0] not in feature_map:
                    # if only has instance with p (or this value have not been checked before)
                    feature_map[feature_value[0]] = 0
            x_train[feature] = x_train[feature].map(feature_map)
            x_test[feature] = x_test[feature].map(feature_map)

        # notice here, because we only train on training set.
        # Then there may have feature values that did not appear in train set.
        # The better way is replacing them by 0.
        return x_train.values, np.nan_to_num(x_test.values)


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
        self.data = MushroomData('mushroom/agaricus-lepiota.data.preprocessing.test.txt')
        self.x, self.y = self.data.get_basic_data()

    def test_get_basic_data(self):
        # size verify
        self.assertTrue(np.shape(self.x) == (10, 20),
                        'Wrong train_test split size')
        # verify label
        expected_y = [0,1,1,0,1,1,1,1,0,1]
        self.assertTrue(all(self.y == expected_y))

    def test_get_scalar_data(self):
        # also verify labels do not in dataset
        x_train, x_test = self.data.get_scalar_data(np.arange(7), np.arange(7,10))
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