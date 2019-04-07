import os
import unittest

from data.abalone_processing import AbaloneData
from data.mushroom_processing import MushroomData
from data.anuran_calls_processing import AnuranCallData
from data.newsgroups_processing import NewsgroupsData

from source import abalone_experiment, anuran_experiment, mushroom_experiment, newsgroups_experiment
from source import data_folder_dir

os.chdir('../')


class TestSource(unittest.TestCase):
    def setUp(self):
        self.data_dir = {
            'abalone': data_folder_dir['abalone'] + AbaloneData.test_filename,
            'mushroom': data_folder_dir['mushroom'] + MushroomData.test_filename,
            'anuran': data_folder_dir['anuran'] + AnuranCallData.test_filename,
            '20news': data_folder_dir['20news'] + NewsgroupsData.test_file_folder
        }

    def test_abalone_experiment(self):
        try:
            abalone_experiment(
                file_dir=self.data_dir['abalone'],
                positive_rings=[5, 6, 7, 8, 9],
                unlabeled_size=0.7)
        except BaseException:
            self.fail('test_abalone_experiment fail')

    def test_anuran_experiment(self):
        try:
            anuran_experiment(file_dir=self.data_dir['anuran'], unlabeled_size=0.7)
        except BaseException:
            self.fail('test_anuran_experiment fail')

    def test_mushroom_experiment(self):
        try:
            mushroom_experiment(file_dir=self.data_dir['mushroom'], unlabeled_size=0.7)
        except BaseException:
            self.fail('test_mushroom_experiment fail')

    def test_newsgroups_experiment(self):
        try:
            newsgroups_experiment(
                folder_dir=self.data_dir['20news'], unlabeled_size=0.7,
                categories=None,
                positive_labels=[
                    'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'
                ],
                feature_score='tfidf', feature_number=100,
                normalize='tfidf', scale=10000.
            )
        except BaseException:
            self.fail('test_newsgroups_experiment fail')
