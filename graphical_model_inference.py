"""
    @nghia nh
    Graphical model inference
    ===
"""

import logging
import unittest
from collections import deque

import numpy as np
from igraph import *


class GraphicalModelWithMST:
    """
    Hard-implement for simple graph with minimum spanning tree like components, binary classification
    """
    def __init__(self, graph_ig):
        logging.info('GraphicalModel __init__')

        self._graph = graph_ig
        self._vertices_num = self._graph.vcount()

        self._component_root = []  # list of component's roots
        self._inference_route = []  # setup as a stack
        self._graph.vs['parent'] = -1
        # igraph treats attribute as only string or numeric value ~.~
        self._vs_message = np.zeros((self._vertices_num, 2))
        self._vs_trace = np.zeros((self._vertices_num, 2), int)

    @property
    def get_label(self):
        return np.array(self._graph.vs['label'])

    @staticmethod
    def factor_calc(receiver_label, weight):
        message = np.array([[-weight * (-1 - receiver_label) ** 2],
                            [-weight * (+1 - receiver_label) ** 2]])
        return message

    def inference(self):
        self._tree_extract()
        self._message_sending()
        self._label_retrieving()

    def _tree_extract(self):
        bfs = deque([])
        free = np.ones(self._vertices_num, bool)
        # consider multi-component graph
        for i in range(self._vertices_num):
            if free[i]:
                free[i] = False
                bfs.append(self._graph.vs[i])
                self._inference_route.append(self._graph.vs[i])
                # set root parent itself makes it easier when dealing root
                self._graph.vs[i]['parent'] = i
                self._component_root.append(i)

                while len(bfs):
                    parent = bfs.popleft()
                    ne_list = parent.neighbors()
                    for ne in ne_list:
                        if free[ne.index]:
                            free[ne.index] = False
                            bfs.append(ne)
                            self._inference_route.append(ne)
                            ne['parent'] = parent.index

    def _message_sending(self):
        for sender in reversed(self._inference_route):
            receiver = self._graph.vs[sender['parent']]
            ne_list = sender.neighbors()

            neighbor_message_sum = np.zeros(2).reshape(2, 1)
            for ne in ne_list:
                if ne != receiver:
                    neighbor_message_sum = neighbor_message_sum + self._vs_message[ne.index].reshape(2, 1)

            # exp and log are annulled
            factor_message = np.array([self.factor_calc(-1, self._graph[sender, receiver]),
                                       self.factor_calc(+1, self._graph[sender, receiver])])
            message = factor_message + neighbor_message_sum

            if sender['label'] > -1:  # verify labeled
                self._vs_trace[sender.index, :] = sender['label']
            else:
                self._vs_trace[sender.index] = np.argmax(message, axis=1).ravel()
            # message.shape = (2,2,1), _vs_trace.shape = (n, 2)
            self._vs_message[sender.index] = message[[0, 1], self._vs_trace[sender.index]].ravel()  # :3

    def _label_retrieving(self):
        for i in self._component_root:
            self._graph.vs[i]['label'] = self._vs_trace[i, 0]
            for trace_back in self._inference_route:
                if trace_back['label'] < 0:
                    parent_label = self._graph.vs[trace_back['parent']]['label']
                    trace_back['label'] = self._vs_trace[trace_back.index, parent_label]


class TestGraphicalModelWithTree(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_vertices(12)
        self.graph.vs['label'] = [-1,1,1,1,0,0,0,0,-1,-1,-1,-1]
        self.graph.add_edges(zip((0,0,2,2,3,4,5,6,7),(6,9,7,10,4,10,6,8,11)))
        self.graph.es['weight'] = [3,5,6,5,3,6,4,3,4]
        self.model = GraphicalModelWithMST(self.graph)
        self.model.inference()

    def test_factor_calc(self):
        self.assertTrue(np.array_equal(GraphicalModelWithMST.factor_calc(1, 2),
                               np.array([-8,0]).reshape(2, 1)))
        self.assertTrue(np.array_equal(GraphicalModelWithMST.factor_calc(-1, 3),
                            np.array([0, -12]).reshape(2, 1)))

    def test_inference_tree_extract(self):
        # check inference route
        expected_route_id = [0,6,9,5,8,1,2,7,10,11,4,3]
        route_id = [i.index for i in self.model._inference_route]
        self.assertEqual(expected_route_id, route_id)

        # check root list
        expected_root_list = [0, 1, 2]
        self.assertEqual(expected_root_list, self.model._component_root)

    def test_inference_message_sending(self):
        # check message sending
        # sending route [3 4 11 10 7 2 1 8 5 9 6 0]
        #
        # mu_3_4 = max( ln(phi_3_4)), label[3] = +1, W[3,4] = 3
        factor = np.array([self.model.factor_calc(-1, 3),
                           self.model.factor_calc(1, 3)])
        mu_3_4 = np.array([factor[0, 1], factor[1, 1]]).reshape(2, 1)
        trace_3_4 = np.array([1, 1]).reshape(2, 1)

        # mu_4_10 = max( ln(phi_4_10) + mu_3_4), label[4] = -1, W[4,10] = 6
        factor = np.array([self.model.factor_calc(-1, 6),
                           self.model.factor_calc(1, 6)])
        message = factor + mu_3_4
        mu_4_10 = np.array([message[0, 0], message[1, 0]]).reshape(2, 1)
        trace_4_10 = np.array([0, 0]).reshape(2, 1)

        # mu_11_7 = max(ln(phi_11_7)), W[11,7] = 4
        factor = np.array([self.model.factor_calc(-1, 4),
                           self.model.factor_calc(1, 4)])
        mu_11_7 = np.max(factor, axis=1).reshape(2, 1)
        trace_11_7 = np.argmax(factor, axis=1).reshape(2, 1)

        # mu_10_2 = max( ln(phi_10_2) + mu_4_10), W[10,2] = 5
        factor = np.array([self.model.factor_calc(-1, 5),
                           self.model.factor_calc(1, 5)])
        message = factor + mu_4_10
        mu_10_2 = np.max(message, axis=1).reshape(2, 1)
        trace_10_2 = np.argmax(message, axis=1).reshape(2, 1)

        # mu_7_2 = max( ln(phi_7_2) + mu_11_7), label[7] = -1, W[7,2] = 6
        factor = np.array([self.model.factor_calc(-1, 6),
                           self.model.factor_calc(1, 6)])
        message = factor + mu_11_7
        mu_7_2 = np.array([message[0, 0], message[1, 0]]).reshape(2, 1)
        trace_7_2 = np.array([0, 0]).reshape(2, 1)

        # mu_2_2 = max( mu_7_2 + mu_10_2), label[2] = +1
        message = mu_7_2 + mu_10_2
        mu_2_2 = np.array([message[1], message[1]]).reshape(2, 1)
        trace_2_2 = np.array([1, 1]).reshape(2, 1)

        # mu_1_1 = max( [0, 0])), label[1] = +1
        factor = np.array([self.model.factor_calc(-1, 0),
                           self.model.factor_calc(1, 0)])
        mu_1_1 = np.array([factor[0, 1], factor[1, 1]]).reshape(2, 1)
        trace_1_1 = np.array([1, 1]).reshape(2, 1)

        # mu_8_6 = max( ln(phi_8_6)), W[8,6] = 3
        factor = np.array([self.model.factor_calc(-1, 3),
                           self.model.factor_calc(1, 3)])
        mu_8_6 = np.max(factor, axis=1).reshape(2, 1)
        trace_8_6 = np.argmax(factor, axis=1).reshape(2, 1)

        # mu_5_6 = max( ln(phi_5_6)), label[5] = -1, W[5,6] = 4
        factor = np.array([self.model.factor_calc(-1, 4),
                           self.model.factor_calc(1, 4)])
        mu_5_6 = np.array([factor[0, 0], factor[1, 0]]).reshape(2, 1)
        trace_5_6 = np.array([0, 0]).reshape(2, 1)

        # mu_9_0 = max( ln(phi_9_0)), W[9,0] = 5
        factor = np.array([self.model.factor_calc(-1, 5),
                           self.model.factor_calc(1, 5)])
        mu_9_0 = np.max(factor, axis=1).reshape(2, 1)
        trace_9_0 = np.argmax(factor, axis=1).reshape(2, 1)

        # mu_6_0 = max( ln(phi_6_0) + mu_5_6 + mu_8_6), label[6] = -1, W[6,0] = 3
        factor = np.array([self.model.factor_calc(-1, 3),
                           self.model.factor_calc(1, 3)])
        message = factor + mu_5_6 + mu_8_6
        mu_6_0 = np.array([factor[0, 0], factor[1, 0]]).reshape(2, 1)
        trace_6_0 = np.array([0, 0]).reshape(2, 1)

        # mu_0_0 = max( mu_6_0 + mu_9_0)
        message = mu_6_0 + mu_9_0
        mu_0_0 = np.array([np.max(message), np.max(message)]).reshape(2, 1)
        trace_0_0 = np.array([np.argmax(message), np.argmax(message)]).reshape(2, 1)

        expected_message = np.array([mu_0_0, mu_1_1, mu_2_2,
                                     mu_3_4, mu_4_10, mu_5_6,
                                     mu_6_0, mu_7_2, mu_8_6,
                                     mu_9_0, mu_10_2, mu_11_7])
        expected_message = expected_message.reshape(self.model._vertices_num, 2)

        expected_trace  = np.array([trace_0_0, trace_1_1, trace_2_2,
                                    trace_3_4, trace_4_10, trace_5_6,
                                    trace_6_0, trace_7_2, trace_8_6,
                                    trace_9_0, trace_10_2, trace_11_7])
        expected_trace = expected_trace.reshape(self.model._vertices_num, 2)

        message = self.model._vs_message
        trace = self.model._vs_trace

        self.assertTrue(np.array_equal(expected_message,message))
        self.assertTrue(np.array_equal(expected_trace, trace))

    def test_get_label(self):
        expected_label = [0,1,1,1,0,0,0,0,0,0,0,0]
        label = self.model.get_label
        self.assertTrue(expected_label == label)