"""
    @nghia nh
    Graph construction
    ---
"""

import unittest
import logging

import numpy as np
from igraph import *


class GraphConstruction:
    def __init__(self, x_l, y_l, x_u, metric='euclidean'):
        logging.info('GraphConstruction __init__, metric=%s', metric)

        self._x = np.concatenate((x_l, x_u), axis=0)
        self._y = np.concatenate((y_l, np.full(len(x_u), -1)))  # unlabeled data have label -1
        self._labeled_data_number = len(x_l)
        self._unlabeled_data_number = len(x_u)

        # init fully connected weighted graph, the edges have not been weighted yet!
        # labeled data indexes from 0..|x_l|-1, unlabeled data from |x_l|..|_x|-1
        self._graph = Graph.Full(len(self._x))
        self._graph.es['weight'] = 0
        self._graph.vs['label'] = self._y  # label vertices

        if metric=='euclidean':
            self.__euclidean()

    @property
    def get_graph(self):
        return self._graph

    @property
    def get_unlabeled_data_number(self):
        return self._unlabeled_data_number

    def __euclidean(self):
        """
        Calc distance metric using euclidean
        :return:
        """
        # TODO matrix calc here
        for i in range(len(self._x)):
            for j in range(i+1, len(self._x)):
                self._graph[i, j] = np.linalg.norm(self._x[i] - self._x[j])
                self._graph[j, i] = self._graph[i, j]

    def construct_knn_graph(self, k=3):
        """
        Construct graph using KNN approach.
        It guarantees that at least 1 nearest neighbor is labeled data.
        :param k: number of nearest neighbors
        :return:
        """
        logging.info('construct_knn_graph with k = %s', k)

        knn_adj_matrix = np.zeros((len(self._x),len(self._x)), dtype=bool)

        # finds k nearest neighbors
        for vertex_id, distance_vector in enumerate(self._graph[:,:]):
            k_min_ids = np.argpartition(distance_vector, k+1)[1:k+1]  # get indices of k minimums values
            # for unlabeled data: must have at least one labeled data as a neighbor
            # -> add closest labeled data
            if vertex_id >= self._labeled_data_number:
                labeled_set = distance_vector[:self._labeled_data_number]
                closest_labeled_data = labeled_set.index(min(labeled_set))
                k_min_ids = np.append(k_min_ids, closest_labeled_data)
            knn_adj_matrix[vertex_id,k_min_ids] = True
        #symmetrize matric
        knn_adj_matrix = np.maximum(knn_adj_matrix, knn_adj_matrix.T)

        # remove unrelated edges
        for vertex_id, row in enumerate(knn_adj_matrix):
            self._graph[vertex_id, np.nonzero(~row)[0]] = 0

    def construct_mst_graph(self):
        """
        Construct graph using minimum spanning tree for each graph's component
        :return:
        """
        logging.info('construct_mst_graph')

        edges_indices = self._graph.spanning_tree(weights=self._graph.es['weight'], return_tree=False)
        mass = np.ones(self._graph.ecount(), dtype=bool)
        mass[edges_indices] = False
        self._graph.delete_edges(np.nonzero(mass)[0])


class TestGraphConstruction(unittest.TestCase):
    # Test data
    # x_l
    x_l = np.array([(1, 1), (2, 3), (3, 5)])
    # y_l
    y_l = np.array([0, 1, 1])
    # x_u
    x_u = np.array([(4, 0), (1, 0)])
    #
    ###
    # x = (1,1),(2,3),(3,5),(4,0),(1,0)

    def setUp(self):
        self.graph_constructor = GraphConstruction(self.x_l, self.y_l, self.x_u, metric='euclidean')
        self.graph = self.graph_constructor.get_graph

    def test_euclidean_distance(self):
        expected_weight_adj_matrix = [[0,       2.236,  4.472,  3.162,  1.0],
                                      [2.236,   0,      2.236,  3.606,  3.162],
                                      [4.472,   2.236,  0,      5.099,  5.385],
                                      [3.162,   3.606,  5.099,  0,      3.0],
                                      [1.0,     3.162,  5.385,  3.0,    0]]
        # verify adj weight matrix
        diff = np.abs(np.array(expected_weight_adj_matrix) - np.asarray(self.graph[:,:]))
        self.assertTrue( all((diff < 0.0001).tolist()) )
        # verify graph label
        expected_y_label = [0, 1, 1, -1, -1]
        self.assertTrue(self.graph.vs['label'] == expected_y_label)

    def test_knn_construction(self):
        self.graph_constructor.construct_knn_graph(k=2)
        expected_weight_adj_matrix = [[0,       2.236,  4.472,  3.162,  1.0],
                                      [2.236,   0,      2.236,  0,      0],
                                      [4.472,   2.236,  0,      0,      0],
                                      [3.162,   0,      0,      0,      3.0],
                                      [1.0,     0,      0,      3.0,    0]]
        # verify adj weight matrix
        diff = np.abs(np.array(expected_weight_adj_matrix) - np.asarray(self.graph[:, :]))
        self.assertTrue(all((diff < 0.0001).tolist()))

    def test_spanning_tree_construction(self):
        self.graph_constructor.construct_mst_graph()
        expected_weight_adj_matrix = [[0,       2.236,  0,      0,      1.0],
                                      [2.236,   0,      2.236,  0,      0],
                                      [4.472,   0,      0,      0,      0],
                                      [0,       0,      0,      0,      3.0],
                                      [1.0,     0,      0,      3.0,    0]]
        # verify adj weight matrix
        diff = np.abs(np.array(expected_weight_adj_matrix) - np.asarray(self.graph[:, :]))
        self.assertTrue(all((diff < 0.0001).tolist()))