import unittest

from igraph import Graph
import numpy as np

from mincut_inference import MincutInference

class TestMincutInference(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_vertices(12)
        self.graph.vs['label'] = [1,1,1,1,0,0,0,0,-1,-1,-1,-1]
        self.graph.add_edges(zip((0,0,0,2,3,3,4,5,6,7),(3,6,9,4,7,10,10,6,8,11)))
        self.graph.es['weight'] = [2,3,5,3,6,5,6,4,3,4]
        self.model = MincutInference(self.graph)
        self.model.inference()

    def test_inference(self):
        expected_positive_part = [0, 1, 2, 3, 9, 13]
        expected_negative_part = [4,5,6,7,8,10,11,12]
        expected_weight = [2,3,5,3,6,5,6,4,3,4]
        # weight
        self.assertTrue(expected_weight == self.model._graph.es['weight'])
        # all positive labeled data in 1st part
        self.assertEqual(expected_positive_part, self.model._cut.partition[0])
        self.assertEqual(expected_negative_part, self.model._cut.partition[1])
        # cut value
        self.assertEqual(self.model._cut.value, 17)

    def test_get_label(self):
        expected_y = np.array([1,1,1,1,0,0,0,0,0,1,0,0])
        self.assertTrue(np.array_equal(expected_y, self.model.labels))