import unittest

from source import data_dir
from data.anuran_calls_processing import AnuranCallData


class TestAnuranCallData(unittest.TestCase):
    def setUp(self):
        self.data = AnuranCallData('../../' + data_dir['anuran'])
        self.data.transform()

    def test_init(self):
        # verify size after init processing
        self.assertTrue(self.data._x.shape == (7195, 22), 'Data size is not equal')
        # verify Sex feature
        expected_y = [1, 1, 1, 1, 1, 1]
        self.assertTrue(all(expected_y == self.data._y[:6]), 'Wrong Family feature')
