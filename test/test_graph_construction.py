import unittest

from graph_construction import GraphConstruction

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

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
        self.graph = self.graph_constructor.graph
        self.euclidean_distance_matrix = np.array([[0, np.sqrt(1 + 4), np.sqrt(4 + 16), np.sqrt(9 + 1), np.sqrt(0 + 1)],
                                                  [np.sqrt(1 + 4), 0, np.sqrt(1 + 4), np.sqrt(4 + 9), np.sqrt(1 + 9)],
                                                  [np.sqrt(4 + 16), np.sqrt(1 + 4), 0, np.sqrt(1 + 25), np.sqrt(4 + 25)],
                                                  [np.sqrt(9 + 1), np.sqrt(4 + 9), np.sqrt(1 + 25), 0, np.sqrt(9 + 0)],
                                                  [np.sqrt(0 + 1), np.sqrt(1 + 9), np.sqrt(4 + 25), np.sqrt(9 + 0), 0]])
        # squared Euclidean
        #  0,  5, 20, 10,  1
        #  5,  0,  5, 13, 10
        # 20,  5,  0, 26, 29
        # 10, 13, 26,  0,  9
        #  1, 10, 29,  9,  0

    def test_euclidean_similarity(self):
        expected_weight = 1/(1 + self.euclidean_distance_matrix)
        di = np.diag_indices(5)
        expected_weight[di] = 0

        adj_matrix = np.array(self.graph.get_adjacency('weight').data)
        self.assertTrue(np.allclose(expected_weight, adj_matrix))

        # verify graph label
        expected_y_label = [0, 1, 1, -1, -1]
        self.assertTrue(self.graph.vs['label'] == expected_y_label)

    def test_rbf_kernel(self):
        self.graph_constructor.re_weight_graph(metric='rbf')
        expected_weight = np.exp(- self.euclidean_distance_matrix**2 / ((1/2)**2))
        di = np.diag_indices(5)
        expected_weight[di] = 0

        adj_matrix = np.array(self.graph.get_adjacency('weight').data)
        self.assertTrue(np.allclose(expected_weight, adj_matrix))

    def test_cosine_similarity(self):
        self.graph_constructor.re_weight_graph(metric='cosine')
        expected_weight = cosine_similarity(self.graph_constructor._x, self.graph_constructor._x)
        di = np.diag_indices(5)
        expected_weight[di] = 0

        adj_matrix = np.array(self.graph.get_adjacency('weight').data)
        self.assertTrue(np.allclose(expected_weight, adj_matrix))

    def test_construct_knn_graph(self):
        # squared Euclidean
        #  0,  5, 20, 10,  1
        #  5,  0,  5, 13, 10
        # 20,  5,  0, 26, 29
        # 10, 13, 26,  0,  9
        #  1, 10, 29,  9,  0
        self.graph_constructor._construct_knn_graph(k=2)
        expected_adj_matrix = np.array([[0, 1, 1, 1, 1],
                                        [1, 0, 1, 0, 0],
                                        [1, 1, 0, 0, 0],
                                        [1, 0, 0, 0, 1],
                                        [1, 0, 0, 1, 0]]).astype(bool)
        adj_matrix = np.array(self.graph.get_adjacency('weight').data).astype(bool)
        self.assertTrue(np.array_equal(expected_adj_matrix, adj_matrix))

    def test_construct_mst_graph(self):
        self.graph_constructor._construct_mst_graph()
        expected_adj_matrix = np.array([[0, 1, 0, 0, 1],
                                        [1, 0, 1, 0, 0],
                                        [0, 1, 0, 0, 0],
                                        [0, 0, 0, 0, 1],
                                        [1, 0, 0, 1, 0]]).astype(bool)
        adj_matrix = np.array(self.graph.get_adjacency('weight').data).astype(bool)
        self.assertTrue(np.array_equal(expected_adj_matrix, adj_matrix))