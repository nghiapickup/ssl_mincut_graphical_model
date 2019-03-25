"""
    @nghia nh
    Preprocessing Abalone data
    ---

    The dataset itselt is vectorized with only sex feature
    need to be normalizing (I=0,M=1,F=2).
    This script only keeps the classes with instances number > 100: class 5-15.
    Also due to binary classifier testing only, there has an option using
    1 designed class, list of classes for positive label and turns others to negative.
"""

import unittest
import logging

import pandas as pd


class AbaloneData:
    default_file_name = 'abalone.data.txt'
    data_header = ['Sex','Length','Diameter','Height','Whole weight',
                   'Shucked weight','Viscera weight','Shell weight','Rings']

    def __init__(self, data_file_dir):
        logging.info('AbaloneData __init__')
        self.abalone_df = pd.read_csv(data_file_dir, names = self.data_header)
        # Scalarize 'Sex' feature
        self.abalone_df['Sex'] = self.abalone_df['Sex'].map({'I':0, 'M':1, 'F':2})
        # Get only data with label range from [5:15]
        self.abalone_df = self.abalone_df[(self.abalone_df['Rings'] >= 5) & (self.abalone_df['Rings'] <= 15)]

    def get_binary_class_data(self, positive_label_list):
        """
        return binary classification data set
        :param positive_label_list: list of label will be set positive
        :return: ndarray pair of (x, y)
        """
        logging.info('AbaloneData getBinaryInput with positive_label_list = %s', positive_label_list)
        # return binary classification dataset
        x = self.abalone_df.drop(columns=['Rings']).values
        y = self.abalone_df['Rings'].isin(positive_label_list).astype('int').values
        return x, y

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
        self.data = AbaloneData('abalone/abalone.data.preprocessing.test.txt')

    def test_init(self):
        # verify size after init processing
        self.assertTrue(self.data.abalone_df.shape == (6,9), 'Data size is not equal')
        # verify Sex feature
        expected_sex = [1, 1, 1, 0, 0, 1]
        self.assertTrue(all(expected_sex == self.data.abalone_df['Sex']), 'Wrong Sex feature scalarizing')
        # verify Rings feature
        expected_rings = [15, 7, 10, 7, 8, 9]
        self.assertTrue(all(expected_rings == self.data.abalone_df['Rings']), 'Wrong Rings feature')

    def test_getBinaryInput(self):
        x, y = self.data.get_binary_class_data([7])
        # size verify
        self.assertTrue((len(x), len(y)) == (6, 6),
                        'Wrong train_test split size')
        # verify label
        expected_y = [0,1,0,1,0,0]
        self.assertTrue(all(y == expected_y))

    def test_getMultiBinaryInput(self):
        # also verify labels do not in dataset
        x, y = self.data.get_binary_class_data([7, 20, 8])
        # size verify
        self.assertTrue((len(x), len(y)) == (6, 6),
                        'Wrong train_test split size')
        # verify label
        expected_y = [0,1,0,1,1,0]
        self.assertTrue(all(y == expected_y))