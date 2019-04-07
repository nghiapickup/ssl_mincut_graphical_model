"""
    @nghia nh
    ---
    Preprocessing Mushroom data
"""

import logging

import pandas as pd
import numpy as np

from data.base_processing import BaseDataProcessor


class MushroomData(BaseDataProcessor):
    """
    All features are categorical data. We only need to convert them into
    a reasonable meaning which can calc the metric. For each unique value of
    one feature, we compute the proportion between edible and poisonous class
    only on training data.

    Class Distribution:
    edible:     4208 (51.8 %)
    poisonous:  3916 (48.2 %)
    total:      8124 instances
    """
    default_filename = 'agaricus-lepiota.data.txt'
    test_filename = 'agaricus-lepiota.data.source.test.txt'
    data_header = ['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises?', 'odor',
              'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
              'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
              'stalk-surface-below-ring', 'stalk-color-above-ring',
              'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
              'ring-type', 'spore-print-color', 'population', 'habitat']

    def __init__(self, file_dir):
        logging.info('MushroomData __init__')

        self._mushroom_df = pd.read_csv(file_dir, names = self.data_header)
        # Map classes feature
        self._mushroom_df['class'] = self._mushroom_df['class'].map({'p':0, 'e':1})
        # Drop one value and missing features (missing rate > 1/4)
        self._mushroom_df = self._mushroom_df.drop(columns=['veil-type', 'stalk-root'])

        BaseDataProcessor.__init__(self)
        self._x_number = len(self._mushroom_df.index)

    def transform(self):
        logging.info('MushroomData transform')

        self._x = self._mushroom_df.drop(columns='class').values
        self._y = self._mushroom_df['class'].values

        return self

    def extract(self, train_indices, test_indices):
        logging.info('MushroomData extract')

        x = self._mushroom_df.iloc[train_indices]
        x_train = x.drop(columns=['class'])
        x_test = self._mushroom_df.iloc[test_indices].drop(columns=['class'])

        # count unique value in each feature by class labels
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

        # There may have features that does not appear in train set.
        # The better way is replacing them by 0.
        return x_train.values, self._y[train_indices], np.nan_to_num(x_test.values), self._y[test_indices]
