import unittest

from igraph import Graph
import numpy as np

from graphical_model_inference import GraphicalModelWithTree
from graphical_model_inference import GraphicalModelWithLoop


class TestGraphicalModelWithTree(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_vertices(12)
        self.graph.vs['label'] = [-1,1,1,1,0,0,0,0,-1,-1,-1,-1]
        self.graph.add_edges(zip((0,0,2,2,3,4,5,6,7),(6,9,7,10,4,10,6,8,11)))
        self.graph.es['weight'] = [3,5,6,5,3,6,4,3,4]
        self.model = GraphicalModelWithTree(self.graph)
        self.model.inference()

    def test_factor_calc(self):
        self.assertTrue(np.array_equal(GraphicalModelWithTree.factor_calc(1, 2),
                               np.array([-8,0])))
        self.assertTrue(np.array_equal(GraphicalModelWithTree.factor_calc(-1, 3),
                            np.array([0, -12])))

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
        mu_3_4 = np.array([factor[0, 1], factor[1, 1]])
        trace_3_4 = np.array([1, 1])

        # mu_4_10 = max( ln(phi_4_10) + mu_3_4), label[4] = -1, W[4,10] = 6
        factor = np.array([self.model.factor_calc(-1, 6),
                           self.model.factor_calc(1, 6)])
        message = factor + mu_3_4
        mu_4_10 = np.array([message[0, 0], message[1, 0]])
        trace_4_10 = np.array([0, 0])

        # mu_11_7 = max(ln(phi_11_7)), W[11,7] = 4
        factor = np.array([self.model.factor_calc(-1, 4),
                           self.model.factor_calc(1, 4)])
        mu_11_7 = np.max(factor, axis=1)
        trace_11_7 = np.argmax(factor, axis=1)

        # mu_10_2 = max( ln(phi_10_2) + mu_4_10), W[10,2] = 5
        factor = np.array([self.model.factor_calc(-1, 5),
                           self.model.factor_calc(1, 5)])
        message = factor + mu_4_10
        mu_10_2 = np.max(message, axis=1)
        trace_10_2 = np.argmax(message, axis=1)

        # mu_7_2 = max( ln(phi_7_2) + mu_11_7), label[7] = -1, W[7,2] = 6
        factor = np.array([self.model.factor_calc(-1, 6),
                           self.model.factor_calc(1, 6)])
        message = factor + mu_11_7
        mu_7_2 = np.array([message[0, 0], message[1, 0]])
        trace_7_2 = np.array([0, 0])

        # mu_2_2 = max( mu_7_2 + mu_10_2), label[2] = +1
        message = mu_7_2 + mu_10_2
        mu_2_2 = np.array([message[1], message[1]])
        trace_2_2 = np.array([1, 1])

        # mu_1_1 = max( [0, 0])), label[1] = +1
        factor = np.array([self.model.factor_calc(-1, 0),
                           self.model.factor_calc(1, 0)])
        mu_1_1 = np.array([factor[0, 1], factor[1, 1]])
        trace_1_1 = np.array([1, 1])

        # mu_8_6 = max( ln(phi_8_6)), W[8,6] = 3
        factor = np.array([self.model.factor_calc(-1, 3),
                           self.model.factor_calc(1, 3)])
        mu_8_6 = np.max(factor, axis=1)
        trace_8_6 = np.argmax(factor, axis=1)

        # mu_5_6 = max( ln(phi_5_6)), label[5] = -1, W[5,6] = 4
        factor = np.array([self.model.factor_calc(-1, 4),
                           self.model.factor_calc(1, 4)])
        mu_5_6 = np.array([factor[0, 0], factor[1, 0]])
        trace_5_6 = np.array([0, 0])

        # mu_9_0 = max( ln(phi_9_0)), W[9,0] = 5
        factor = np.array([self.model.factor_calc(-1, 5),
                           self.model.factor_calc(1, 5)])
        mu_9_0 = np.max(factor, axis=1)
        trace_9_0 = np.argmax(factor, axis=1)

        # mu_6_0 = max( ln(phi_6_0) + mu_5_6 + mu_8_6), label[6] = -1, W[6,0] = 3
        factor = np.array([self.model.factor_calc(-1, 3),
                           self.model.factor_calc(1, 3)])
        message = factor + mu_5_6 + mu_8_6
        mu_6_0 = np.array([factor[0, 0], factor[1, 0]])
        trace_6_0 = np.array([0, 0])

        # mu_0_0 = max( mu_6_0 + mu_9_0)
        message = mu_6_0 + mu_9_0
        mu_0_0 = np.array([np.max(message), np.max(message)])
        trace_0_0 = np.array([np.argmax(message), np.argmax(message)])

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

        message = self.model._messages
        trace = self.model._trace

        self.assertTrue(np.array_equal(expected_message,message))
        self.assertTrue(np.array_equal(expected_trace, trace))

    def test_get_label(self):
        expected_label = np.array([0,1,1,1,0,0,0,0,0,0,0,0])
        label = self.model.labels
        self.assertTrue(np.array_equal(expected_label, label))


class TestGraphicalModelWithLoop(unittest.TestCase):
    def setUp(self):
        self.graph = Graph()
        self.graph.add_vertices(6)
        self.graph.vs['label'] = [1, 1, -1, -1, 0, -1]
        self.graph.add_edges(zip((0, 0, 1, 1, 2, 3, 4), (1, 3, 2, 5, 3, 4, 5)))
        self.graph.es['weight'] = [3, 5, 6, 3, 4, 2, 5]
        self.model = GraphicalModelWithLoop(self.graph)
        self.model.inference()

    def test__send_flooding_message(self):

        # indices   0       1       2       3       4       5       6
        # edges     (0,1)   (0,3)   (1,2)   (1,5)   (2,3)   (3,4)   (4,5)
        # weight    3       5       6       3       4       2       5
        # factor func:  exp[-w[u,v] * [f(u) - f(v)]**2]

        # label     1       1       -1      -1      0       -1

        #init messages
        messages = np.zeros((self.graph.ecount(), 2, 2))

        # W[0, 1] =3, label[0]=+1, label[1]=+1
        # source:0 -> target:1
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # target=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # target=1
        ])  # source's label = +1
        messages[0, 0] = factor[:, 1]

        # target:1 -> source:0
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # source=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # source=1
        ])
        messages[0, 1] = factor[:, 1]  # target's label = +1

        # W[0, 3] =5, label[0]=+1
        # source:0 -> target:3
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # target=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # target=1
        ])
        messages[1, 0] = factor[:, 1]  # source's label = +1

        # target:3 -> source:0
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # source=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # source=1
        ])
        messages[1, 1] = factor.max(axis=1)

        # W[1, 2] =6, label[1]=+1
        # source:1 -> target:2
        factor = np.array([
            np.array([-6 * (-1 - -1) ** 2, -6 * (1 - -1) ** 2]),  # target=-1
            np.array([-6 * (-1 - 1) ** 2, -6 * (1 - 1) ** 2])  # target=1
        ])
        messages[2, 0] = factor[:, 1]  # source's label = +1
        # target:2 -> source:1
        factor = np.array([
            np.array([-6 * (-1 - -1) ** 2, -6 * (1 - -1) ** 2]),  # source=-1
            np.array([-6 * (-1 - 1) ** 2, -6 * (1 - 1) ** 2])  # source=1
        ])
        messages[2, 1] = factor.max(axis=1)

        # W[1, 5] =3, label[1]=+1
        # source:1 -> target:5
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # target=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # target=1
        ])
        messages[3, 0] = factor[:, 1]  # source's label = +1
        # target:5 -> source:1
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # source=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # source=1
        ])
        messages[3, 1] = factor.max(axis=1)

        # W[2, 3] =4
        # source:2 -> target:3
        factor = np.array([
            np.array([-4 * (-1 - -1) ** 2, -4 * (1 - -1) ** 2]),  # target=-1
            np.array([-4 * (-1 - 1) ** 2, -4 * (1 - 1) ** 2])  # target=1
        ])
        messages[4, 0] = factor.max(axis=1)
        # target:3 -> source:2
        factor = np.array([
            np.array([-4 * (-1 - -1) ** 2, -4 * (1 - -1) ** 2]),  # source=-1
            np.array([-4 * (-1 - 1) ** 2, -4 * (1 - 1) ** 2])  # source=1
        ])
        messages[4, 1] = factor.max(axis=1)

        # W[3, 4] =2, label[4]=-1
        # source:3 -> target:4
        factor = np.array([
            np.array([-2 * (-1 - -1) ** 2, -2 * (1 - -1) ** 2]),  # target=-1
            np.array([-2 * (-1 - 1) ** 2, -2 * (1 - 1) ** 2])  # target=1
        ])
        messages[5, 0] = factor.max(axis=1)
        # target:4 -> source:3
        factor = np.array([
            np.array([-2 * (-1 - -1) ** 2, -2 * (1 - -1) ** 2]),  # source=-1
            np.array([-2 * (-1 - 1) ** 2, -2 * (1 - 1) ** 2])  # source=1
        ])
        messages[5, 1] = factor[:, 0]  # target's label = -1

        # W[4, 5] =5, label[4]=-1
        # source:4 -> target:5
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # target=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # target=1
        ])
        messages[6, 0] = factor[:, 0]  # source's label = -1
        # target:5 -> source:4
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # source=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # source=1
        ])
        messages[6, 1] = factor.max(axis=1)

        # indices   0       1       2       3       4       5       6
        # edges     (0,1)   (0,3)   (1,2)   (1,5)   (2,3)   (3,4)   (4,5)
        # weight    3       5       6       3       4       2       5
        # factor func:  exp[-w[u,v] * [f(u) - f(v)]**2]

        # label     1       1       -1      -1      0       -1

        # assert first init messages
        self.model._messages = np.zeros((self.graph.ecount(), 2, 2))
        self.graph.vs['label'] = [1, 1, -1, -1, 0, -1]
        self.model._send_flooding_message(threshold=0)
        self.assertTrue(np.array_equal(self.model._messages, messages))

        # Sending 1st messages

        first_messages = np.zeros((self.graph.ecount(), 2, 2))

        # W[0, 1] =3, label[0]=+1, label[1]=+1

        # source:0 -> target:1
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # target=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # target=1
        ])  # source's label = +1
        # neighbor: (0, 3):1
        ne_sum = messages[1, 1]
        first_messages[0, 0] = (ne_sum + factor)[:, 1]

        # target:1 -> source:0
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # source=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (1, 2):2 (1, 5):3
        ne_sum = messages[2, 1] + messages[3, 1]
        first_messages[0, 1] = (ne_sum + factor)[:, 1]  # target's label = +1

        # W[0, 3] =5, label[0]=+1

        # source:0 -> target:3
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # target=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (0, 1): 0
        ne_sum = messages[0, 1]
        first_messages[1, 0] = (ne_sum + factor)[:, 1]  # source's label = +1

        # target:3 -> source:0
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # source=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (2, 3):4  (3, 4):5
        ne_sum = messages[4, 0] + messages[5, 1]
        first_messages[1, 1] = (ne_sum + factor).max(axis=1)

        # W[1, 2] =6, label[1]=+1

        # source:1 -> target:2
        factor = np.array([
            np.array([-6 * (-1 - -1) ** 2, -6 * (1 - -1) ** 2]),  # target=-1
            np.array([-6 * (-1 - 1) ** 2, -6 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (0, 1):0 (1, 5):3
        ne_sum = messages[0, 0] + messages[3, 1]
        first_messages[2, 0] = (ne_sum + factor)[:, 1]  # source's label = +1

        # target:2 -> source:1
        factor = np.array([
            np.array([-6 * (-1 - -1) ** 2, -6 * (1 - -1) ** 2]),  # source=-1
            np.array([-6 * (-1 - 1) ** 2, -6 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (2, 3):4
        ne_sum = messages[4, 1]
        first_messages[2, 1] = (ne_sum + factor).max(axis=1)

        # W[1, 5] =3, label[1]=+1

        # source:1 -> target:5
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # target=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (0, 1):0 (1, 2):2
        ne_sum = messages[0, 0] + messages[2, 1]
        first_messages[3, 0] = (ne_sum + factor)[:, 1]  # source's label = +1

        # target:5 -> source:1
        factor = np.array([
            np.array([-3 * (-1 - -1) ** 2, -3 * (1 - -1) ** 2]),  # source=-1
            np.array([-3 * (-1 - 1) ** 2, -3 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (4, 5):6
        ne_sum = messages[6, 0]
        first_messages[3, 1] = (ne_sum + factor).max(axis=1)

        # W[2, 3] =4

        # source:2 -> target:3
        factor = np.array([
            np.array([-4 * (-1 - -1) ** 2, -4 * (1 - -1) ** 2]),  # target=-1
            np.array([-4 * (-1 - 1) ** 2, -4 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (1, 2):2
        ne_sum = messages[2, 0]
        first_messages[4, 0] = (ne_sum + factor).max(axis=1)

        # target:3 -> source:2
        factor = np.array([
            np.array([-4 * (-1 - -1) ** 2, -4 * (1 - -1) ** 2]),  # source=-1
            np.array([-4 * (-1 - 1) ** 2, -4 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (0, 3):1 (3, 4):5
        ne_sum = messages[1, 0] + messages[5, 1]
        first_messages[4, 1] = (ne_sum + factor).max(axis=1)

        # W[3, 4] =2, label[4]=-1

        # source:3 -> target:4
        factor = np.array([
            np.array([-2 * (-1 - -1) ** 2, -2 * (1 - -1) ** 2]),  # target=-1
            np.array([-2 * (-1 - 1) ** 2, -2 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (0, 3):1 (2, 3): 4
        ne_sum = messages[1, 0] + messages[4, 0]
        first_messages[5, 0] = (ne_sum + factor).max(axis=1)

        # target:4 -> source:3
        factor = np.array([
            np.array([-2 * (-1 - -1) ** 2, -2 * (1 - -1) ** 2]),  # source=-1
            np.array([-2 * (-1 - 1) ** 2, -2 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (4, 5): 6
        ne_sum = messages[6, 1]
        first_messages[5, 1] = (ne_sum + factor)[:, 0]  # target's label = -1

        # W[4, 5] =5, label[4]=-1

        # source:4 -> target:5
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # target=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # target=1
        ])
        # neighbor: (3, 4): 5
        ne_sum = messages[5, 0]
        first_messages[6, 0] = (ne_sum + factor)[:, 0]  # source's label = -1

        # target:5 -> source:4
        factor = np.array([
            np.array([-5 * (-1 - -1) ** 2, -5 * (1 - -1) ** 2]),  # source=-1
            np.array([-5 * (-1 - 1) ** 2, -5 * (1 - 1) ** 2])  # source=1
        ])
        # neighbor: (1, 5): 3
        ne_sum = messages[3, 0]
        first_messages[6, 1] = (ne_sum + factor).max(axis=1)

        # assert first messages sending
        self.model._messages = np.zeros((self.graph.ecount(), 2, 2))
        self.graph.vs['label'] = [1, 1, -1, -1, 0, -1]
        self.model._send_flooding_message(threshold=1)
        self.assertTrue(np.array_equal(self.model._messages, first_messages))

    def test_retrieve_label(self):
        expected_label = np.array([1, 1, 1, 1, 0, 0])
        label = self.model.labels
        self.assertTrue(np.array_equal(expected_label, label))
