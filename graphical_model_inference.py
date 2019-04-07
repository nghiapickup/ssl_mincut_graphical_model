"""
    @nghia nh
    ---
    Graphical model inference

"""

import logging
from collections import deque

import numpy as np


class GraphicalModelWithTree:
    """
    Hard-implement for simple graphical model inference
    with tree like components graph, binary classification
    """
    def __init__(self, i_graph):
        """
        :param i_graph: input graph as Graph instance of igraph
        """
        logging.info('GraphicalModelWithTree __init__')

        self._graph = i_graph
        self._vertices_num = self._graph.vcount()

        self._component_root = []  # list of component's roots
        self._inference_route = []  # setup as a stack
        self._graph.vs['parent'] = -1
        # igraph treats attribute as only string or numeric value ~.~
        self._vs_message = np.zeros((self._vertices_num, 2))
        self._vs_trace = np.zeros((self._vertices_num, 2), int)

    @property
    def labels(self):
        return np.array(self._graph.vs['label'])

    @staticmethod
    def factor_calc(receiver_label, weight):
        message = np.array([[-weight * (-1 - receiver_label) ** 2],
                            [-weight * (+1 - receiver_label) ** 2]])
        return message

    def inference(self):
        """
        inference process
        :return:
        """
        logging.info('GraphicalModelWithTree inference')

        self._tree_extract()
        self._message_sending()
        self._label_retrieving()

    def _tree_extract(self):
        """
        Extract graph input to tree search list (from leaf to root)
        :return:
        """
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
        """
        sending message from leaf to root and save the trace
        :return:
        """
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
        """
        retrieve label from trace
        :return:
        """
        for i in self._component_root:
            self._graph.vs[i]['label'] = self._vs_trace[i, 0]
            for trace_back in self._inference_route:
                if trace_back['label'] < 0:
                    parent_label = self._graph.vs[trace_back['parent']]['label']
                    trace_back['label'] = self._vs_trace[trace_back.index, parent_label]
