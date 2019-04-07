"""
    @nghia nh
    ---
    Preprocessing Anuran Calls (MFCCs) data
"""

import logging
import pandas as pd
from data.base_processing import BaseDataProcessor


class AnuranCallData(BaseDataProcessor):
    """
    This is quite standard format dataset with normalized features
    Then this script only constructs Family label for binary classification task.
    """
    default_filename = 'Frogs_MFCCs.csv'
    test_filename = 'Frogs_MFCCs.source.test.csv'

    def __init__(self, file_dir):
        logging.info('Anuran Calls (MFCCs) __init__')

        self._anuran_df = pd.read_csv(file_dir)
        # Remove classes Genus, Species, and RecordID
        self._anuran_df = self._anuran_df.drop(columns=['Genus', 'Species', 'RecordID'])
        # Map label value for Family
        self._anuran_df['Family'] = self._anuran_df['Family'].\
            map({'Leptodactylidae':1, 'Hylidae':0, 'Dendrobatidae':0, 'Bufonidae':0})

        BaseDataProcessor.__init__(self)
        self._x_number = len(self._anuran_df.index)

    def transform(self):
        logging.info('AnuranCallData transform binary with positive label = Leptodactylidae')

        # return binary classification dataset
        self._x = self._anuran_df.drop(columns=['Family']).values
        self._y = self._anuran_df['Family'].values

        return self
