"""
    @nghia nh
    ---
    Graph construction

"""

import logging

import numpy as np
from igraph import Graph
from scipy.spatial.distance import cosine


class GraphConstruction:
    def __init__(self, x_l, y_l, x_u, metric='euclidean', metric_param=None):
        """
        Construct a basic fully connected graph with input metric
        :param x_l: labeled data
        :param y_l: labeled labels
        :param x_u: unlabeled data
        :param metric: graph weights' metric
        :param metric_param: parameter for metric
        """
        logging.info('GraphConstruction __init__, metric=%s', metric)

        self._x = np.concatenate((x_l, x_u), axis=0)
        self._y = np.concatenate((y_l, np.full(len(x_u), -1)))  # unlabeled data have label -1
        self._labeled_data_number = len(x_l)
        self._x_u_number = len(x_u)

        # init fully connected graph
        # labeled data indexes from 0..|x_l|-1, unlabeled data from |x_l|..|_x|-1
        self._graph = Graph.Full(len(self._x))
        self._graph.es['weight'] = 0
        self._graph.vs['label'] = self._y  # label vertices

        # init weight with input metric
        self._metric_map = {
            'euclidean': self._euclidean_similarity,
            'rbf': self._rbf_kernel,
            'cosine': self._cosine_similarity
        }
        self.re_weight_graph(metric, metric_param)

        self._construction_map ={
            'knn': self._construct_knn_graph,
            'mst': self._construct_mst_graph,
            'knn_mst': self._construct_knn_mst_graph
        }

    @property
    def graph(self):
        return self._graph

    @property
    def x_u_number(self):
        return self._x_u_number

    def construct_graph(self, graph_type, **kwargs):
        try:
            self._construction_map[graph_type](**kwargs)
        except KeyError:
            logging.error('GraphConstruction: Non supported graph_type %s.' % graph_type)
            raise

    def _euclidean_similarity(self, metric_param=None):
        """
        Calc similarity metric using euclidean.
        Constrain W[i,i] = 0
        :return:
        """
        # TODO matrix calc here
        for i in range(len(self._x)):
            for j in range(i+1, len(self._x)):
                self._graph[i, j] = 1/(1 + np.linalg.norm(self._x[i] - self._x[j]))
                self._graph[j, i] = self._graph[i, j]

    def _rbf_kernel(self, metric_param=None):
        """
        Calc RBF kernel
        Constrain W[i,i] = 0
        The default bandwidth(None input) is 1/#feature
        :param bandwidth: bandwidth parameter
        :return:
        """
        bandwidth = metric_param
        if bandwidth is None:
            bandwidth = 1/self._x.shape[1]

        # xxx igraph, i want to set by adj matrix
        for i in range(len(self._x)):
            for j in range(i + 1, len(self._x)):
                self._graph[i, j] = np.exp(-np.linalg.norm(self._x[i] - self._x[j])**2 / (bandwidth**2))
                self._graph[j, i] = self._graph[i, j]

    def _cosine_similarity(self, metric_param=None):
        """
        Calc cosine similarity
        Constrain W[i,i] = 0
        :return:
        """
        for i in range(len(self._x)):
            for j in range(i + 1, len(self._x)):
                self._graph[i, j] = 1 - cosine(self._x[i], self._x[j])
                self._graph[j, i] = self._graph[i, j]

    def re_weight_graph(self, metric='euclidean', metric_param=None):
        """
        re-weighting graph
        :param metric: new metric calculator
        :param metric_param: metric's parameter
        :return:
        """
        try:
            self._metric_map[metric](metric_param)
        except KeyError:
            logging.error('GraphConstruction: None supported metric %s.' % metric)
            raise

    def _construct_knn_graph(self, k=3):
        """
        Construct graph using KNN approach (K most similar neighbors!)
        It guarantees that at least 1 selected neighbor is labeled data.
        :param k: number of neighbors, default=3
        :return:
        """
        logging.info('construct_knn_graph with k = %s', k)
        assert k > 0, "k must be a positive int"

        knn_adj_matrix = np.zeros((len(self._x),len(self._x)), dtype=bool)
        adj_matrix = np.array(self._graph.get_adjacency('weight').data)

        # find k most similar neighbors
        # -> get indices of k maximum values on each row
        # exclude the edge itself (has maximum similarity = 1)
        k_similar_ids = np.argpartition(adj_matrix, -k, axis=1)[:, -k:]

        # find most similar labeled neighbor
        labeled_edges_weight = adj_matrix[:, :self._labeled_data_number]
        most_similar_ids = np.argmax(labeled_edges_weight, axis=1)

        # remove non-neighbors
        # TODO find better way
        for index, _ in enumerate(knn_adj_matrix):
            knn_adj_matrix[index, k_similar_ids[index]] = True
            knn_adj_matrix[index, most_similar_ids[index]] = True

        # symmetrize matric
        knn_adj_matrix = np.maximum(knn_adj_matrix, knn_adj_matrix.T)

        # remove unrelated edges
        for vertex_id, row in enumerate(knn_adj_matrix):
            self._graph[vertex_id, np.nonzero(~row)[0]] = 0

    def _construct_mst_graph(self, **kwargs):
        """
        Construct graph using maximum spanning tree for each graph's component
        :return:
        """
        logging.info('construct_mst_graph')

        converted_weight = 1/(1 + np.array(self._graph.es['weight']))
        # find minimum spanning tree on converted weight
        edges_indices = self._graph.spanning_tree(weights=converted_weight, return_tree=False)
        mass = np.ones(self._graph.ecount(), dtype=bool)
        mass[edges_indices] = False
        self._graph.delete_edges(np.nonzero(mass)[0])

    def _construct_knn_mst_graph(self, k=3):
        """
        Construct MST graph from an init KNN graph
        (K most similar neighbors and maximum spanning tree)
        It guarantees that at least 1 selected neighbor is labeled data.
        :param k: number of neighbors, default=3
        :return:
        """
        logging.info('construct_knn_mst_graph with k = %s', k)
        assert k > 0, "k must be a positive int"
        self._construct_knn_graph(k=k)
        self._construct_mst_graph()
