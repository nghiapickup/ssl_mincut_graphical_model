"""
    @nghia nh
    ---
    Mincut approach inference

"""

import logging
import copy

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
        logging.info('Mincut inference')

        self._cut = self._graph.st_mincut(source=self._v_plus, target=self._v_minus, capacity='weight')
        # set label
        [positive_label, _] = self._cut.partition
        positive_label = set(positive_label) - {self._v_plus}
        self._graph.delete_vertices([self._v_minus, self._v_plus])
        y = np.zeros(self._graph.vcount())  # re-construct y
        y[list(positive_label)] = 1
        self._graph.vs['label'] = y


class RandomizeMincutInference:
    def __init__(self, i_graph):
        """
        Setup graph for mincut: add 2 more pseudo vertices and related edges
        :param i_graph: input graph as Graph instance of igraph
        """
        logging.info('RandomizeMincutInference __init__')

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
        inference randomization mincut on input graph
        :return:
        """
        logging.info('RandomizeMincut inference')

        perturbation_num = 4
        noise_rate = 0.3
        noise_range = (-0.6, 0.4)
        base_adj_matrix = copy.deepcopy(self._graph.get_adjacency('weight').data)
        base_adj_matrix = np.array(base_adj_matrix)[:-2, :-2]  # remove pseudo vertices
        vertices_number = self._graph.vcount()
        y = [] # list of return results

        # basic mincut first
        self._cut = self._graph.st_mincut(source=self._v_plus, target=self._v_minus, capacity='weight')
        [positive_label, _] = self._cut.partition

        # ignore this cut if it has unbalance separation
        if len(positive_label) > 0.06 * vertices_number or \
                len(positive_label) < 0.94 * vertices_number:
            labels = np.zeros(vertices_number)  # re-construct y
            labels[positive_label] = 1
            y.append(labels[:-1])  # omit v_plus

        for _ in range(perturbation_num):
            # init noise
            noise_decision = np.random.binomial(
                n=1,
                p=noise_rate,
                size=base_adj_matrix.shape
            )
            noise_value = np.random.uniform(
                low=noise_range[0],
                high=noise_range[1],
                size=base_adj_matrix.shape
            )
            noise_value = noise_decision * noise_value

            # add noise
            noise_weight = np.triu(base_adj_matrix + noise_value)  # only get upper triangle
            self._graph.es['weight'] = noise_weight[noise_weight.nonzero()]

            # find mincut
            self._cut = self._graph.st_mincut(source=self._v_plus, target=self._v_minus, capacity='weight')
            [positive_label, _] = self._cut.partition

            # ignore this cut if it has unbalance separation
            if len(positive_label) < 0.06*vertices_number or \
                    len(positive_label) > 0.94*vertices_number:
                continue

            labels = np.zeros(vertices_number)  # re-construct y
            labels[positive_label] = 1
            y.append(labels[:-1])  # omit v_plus

        self._graph.delete_vertices([self._v_minus, self._v_plus])
        vote = np.array(y).sum(axis=0)  # if sum > cases//2: label 1, else label 0
        self._graph.vs['label'] = (vote > len(y)//2).astype(int)
