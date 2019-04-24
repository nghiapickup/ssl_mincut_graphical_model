import unittest
from io import StringIO

import numpy as np

from source import data_dir, ExperimentSetUp
from data.abalone_processing import AbaloneData


class TestSource(unittest.TestCase):

    def test_experiment(self):
        # test on small amount of Abalone data
        abalone = np.loadtxt(
            '../' + data_dir['abalone'], dtype='str',
            delimiter=',', max_rows=50
        ).reshape((50, 9))

        with StringIO() as input_stream:
            np.savetxt(input_stream, abalone, fmt='%s', delimiter=',')
            input_stream.seek(0)
            data_processor = AbaloneData(
                file_dir=input_stream,
                positive_labels=[5, 6, 7, 8, 9]
            )

        result_file_name = 'abalone_experiment.out'
        exp = ExperimentSetUp(data_processor, result_file_name)
        exp.process_ssl(unlabeled_size=0.5)
        exp.process_sl(test_size=0.5, unlabeled_size=0.5)

        self.assertTrue(True)
