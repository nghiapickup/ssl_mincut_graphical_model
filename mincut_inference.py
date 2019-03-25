"""
    @nghia nh
    Mincut approach inference
    ---
"""

import logging
import unittest

import numpy as np
from igraph import *


class MincutInference:
    def __init__(self, i_graph):
        """
        Setup graph for mincut: add 2 more pseudo vertices and related edges
        :param i_graph: input graph as Graph instance of igraph
        """
        logging.info('MincutInference __init__')

        self._graph = i_graph
        self._cut = None
        self._graph.add_vertices(2)  # add v+ and v-

        # create v+ and v- vertices and connect with corresponding labeled data label
        self._v_plus = self._graph.vcount() - 1
        self._v_minus = self._v_plus - 1
        positive_label_vertices = self._graph.vs.select(label_eq=1).indices
        negative_label_vertices = self._graph.vs.select(label_eq=0).indices
        for label in positive_label_vertices:
            self._graph.add_edge(self._v_plus, label, weight=np.inf)
        for label in negative_label_vertices:
            self._graph.add_edge(self._v_minus, label, weight=np.inf)

    @property
    def get_label(self):
        return np.array(self._graph.vs['label'])

    @property
    def get_cut(self):
        return self._cut

    def inference(self):
        """
        inference mincut on input graph
        :return:
        """
        self._cut = self._graph.st_mincut(source=self._v_plus, target=self._v_minus, capacity='weight')
        self._graph.delete_vertices([self._v_minus, self._v_plus])
        # set label
        [positive_label, _] = self._cut.partition
        positive_label = set(positive_label) - {self._v_plus}
        y = np.zeros(self._graph.vcount())  # re-construct y
        y[list(positive_label)] = 1
        self._graph.vs['label'] = y


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
        self.assertEqual(expected_positive_part, self.model.get_cut.partition[0])
        self.assertEqual(expected_negative_part, self.model.get_cut.partition[1])
        # cut value
        self.assertEqual(self.model.get_cut.value, 17)

    def test_get_label(self):
        expected_y = np.array([1,1,1,1,0,0,0,0,0,1,0,0])
        self.assertTrue(np.array_equal(expected_y, self.model.get_label))