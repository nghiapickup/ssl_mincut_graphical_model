import unittest

from source import data_dir
from data.newsgroups_processing import NewsgroupsData


class TestNewsgroupsData(unittest.TestCase):
    def test_NewsgroupsData(self):
        data = NewsgroupsData(
            folder_dir='../../' + data_dir['20news'],
            categories=None,
            positive_labels=[
                'comp.graphics', 'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x'
            ],
            feature_score='tfidf',
            feature_number=50,
            normalize='tfidf',
        )
        data.transform()
        # Make sure data after transform remain the same number
        self.assertEqual(len(data._x), len(data._y))
