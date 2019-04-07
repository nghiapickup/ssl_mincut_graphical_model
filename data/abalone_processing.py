"""
    @nghia nh
    ---
    Preprocessing Abalone data
"""

import logging
import pandas as pd
from data.base_processing import BaseDataProcessor


class AbaloneData(BaseDataProcessor):
    """
    Binary classification task.
    We only need to vectorize sex feature (I=0,M=1,F=2).
    This script only keeps the classes with instances number > 100: class 5-15.

    Class distribution(training set)
    5	115
    6	259
    7	391
    8	568
    9	689
    10	634
    11	487
    12	267
    13	203
    14	126
    15	103
    """
    default_filename = 'abalone.data.txt'
    test_filename = 'abalone.data.source.test.txt'
    data_header = ['Sex','Length','Diameter','Height','Whole weight',
                   'Shucked weight','Viscera weight','Shell weight','Rings']

    def __init__(self, file_dir, positive_labels=None):
        logging.info('AbaloneData __init__')

        self._abalone_df = pd.read_csv(file_dir, names = self.data_header)
        # 'Sex' feature
        self._abalone_df['Sex'] = self._abalone_df['Sex'].map({'I':0, 'M':1, 'F':2})
        # Get only data with label range from [5:15]
        self._abalone_df = self._abalone_df[(self._abalone_df['Rings'] >= 5) & (self._abalone_df['Rings'] <= 15)]

        BaseDataProcessor.__init__(self)
        self.positive_labels = positive_labels
        self._x_number = len(self._abalone_df.index)

    def transform(self):
        logging.info(
            'AbaloneData transform binary with positive_label_list = %s',
            self.positive_labels)

        assert isinstance(self.positive_labels, list), \
            "positive_label must be defied before transform to binary classification"

        # return binary classification dataset
        self._x = self._abalone_df.drop(columns=['Rings']).values
        self._y = self._abalone_df['Rings'].isin(self.positive_labels).astype('int').values

        return self
