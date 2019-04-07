"""
    @nghia nh
    ---
    Mincut approach inference

"""

import logging
import numpy as np


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
    def labels(self):
        return np.array(self._graph.vs['label'])

    def inference(self):
        """
        inference mincut on input graph
        :return:
        """
        logging.info('MincutInference inference')

        self._cut = self._graph.st_mincut(source=self._v_plus, target=self._v_minus, capacity='weight')
        self._graph.delete_vertices([self._v_minus, self._v_plus])
        # set label
        [positive_label, _] = self._cut.partition
        positive_label = set(positive_label) - {self._v_plus}
        y = np.zeros(self._graph.vcount())  # re-construct y
        y[list(positive_label)] = 1
        self._graph.vs['label'] = y
