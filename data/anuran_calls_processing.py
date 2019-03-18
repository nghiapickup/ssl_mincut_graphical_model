"""
    @nghia nh
    Preprocessing Anuran Calls (MFCCs) data
    ===
    This is quite standard format dataset with normalized values
    Then this script only construct Family label classification data.
"""

import unittest
import logging
import pandas as pd


class AnuranCallData:
    default_file_name = 'Frogs_MFCCs.csv'

    def __init__(self, data_file_dir):
        logging.info('Anuran Calls (MFCCs) __init__')
        self.anuran_df = pd.read_csv(data_file_dir)
        # Remove classes Genus, Species, and RecordID
        self.anuran_df = self.anuran_df.drop(columns=['Genus', 'Species', 'RecordID'])
        # Map label value for Family
        self.anuran_df['Family'] = self.anuran_df['Family'].\
            map({'Leptodactylidae':1, 'Hylidae':0, 'Dendrobatidae':0, 'Bufonidae':0})

    def get_binary_class_data(self):
        logging.info('AnuranCallData getBinaryInput with positive label = Leptodactylidae')
        # return binary classification dataset
        x = self.anuran_df.drop(columns=['Family']).values
        y = self.anuran_df['Family'].values
        return x, y

class TestAbaloneData(unittest.TestCase):

    def setUp(self):
        self.data = AnuranCallData('anuran_calls/Frogs_MFCCs.csv')
        self.x, self.y = self.data.get_binary_class_data()

    def test_init(self):
        # verify size after init processing
        self.assertTrue(self.x.shape == (7195, 22), 'Data size is not equal')
        # verify Sex feature
        expected_y = [1, 1, 1, 1, 1, 1]
        self.assertTrue(all(expected_y == self.y[:6]), 'Wrong Family feature')