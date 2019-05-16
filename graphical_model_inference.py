"""
    @nghia nh
    ---
    Graphical model inference

"""

import logging
from collections import deque

import numpy as np


class GraphicalModelInference:
    """
    Graphical model inference interface
    """
    def __init__(self, i_graph):
        self._graph = i_graph
        self._vertices_num = self._graph.vcount()
        self._edges_num = self._graph.ecount()

    @property
    def labels(self):
        return np.array(self._graph.vs['label'])

    @staticmethod
    def factor_calc(receiver_label, weight, simplified=True):
        """
        Calc factor function with given receiver
        :param receiver_label: label of receiver vertex
        :param weight: W[sender, receiver]
        :param simplified: whether to omit exp calc
        :return:
        """
        message = np.array([
            -weight * (-1 - receiver_label) ** 2,
            -weight * (+1 - receiver_label) ** 2
        ])
        if not simplified:
            message = np.log(message)

        # return shape = (2,)
        return message

    def inference(self):
        """
        inference label for unlabeled label.
        The pipeline only uses this method for it's step.
        :return:
        """
        pass


class GraphicalModelWithTree(GraphicalModelInference):
    """
    Hard-implement for simple graphical model inference
    with tree like components graph, binary classification.
    """
    def __init__(self, i_graph):
        """
        :param i_graph: input graph as Graph instance of igraph
        """
        logging.info('GraphicalModelWithTree __init__')

        GraphicalModelInference.__init__(self, i_graph)
        self._component_root = []  # list of component's roots
        self._inference_route = []  # setup as a stack
        self._graph.vs['parent'] = -1
        # igraph treats attribute as only string or numeric value ~.~
        self._messages = np.zeros((self._vertices_num, 2)) # self._messages[i] message from source i
        self._trace = np.zeros((self._vertices_num, 2), int)

    def inference(self):
        """
        inference process
        :return:
        """
        logging.info('GraphicalModelWithTree inference')

        self._extract_tree()
        self._send_message()
        self._retrieve_label()

    def _extract_tree(self):
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

    def _send_message(self):
        """
        send message from leaf to root and save the trace
        :return:
        """
        for sender in reversed(self._inference_route):
            receiver = self._graph.vs[sender['parent']]
            nes = sender.neighbors()

            neighbor_message_sum = np.zeros(2)
            for ne in nes:
                if ne != receiver:
                    neighbor_message_sum = neighbor_message_sum + self._messages[ne.index]

            # exp and log are annulled
            # factor_message.shape = (2,2)
            factor_message = np.array([self.factor_calc(-1, self._graph[sender, receiver]),
                                       self.factor_calc(+1, self._graph[sender, receiver])])
            # message = [factor_message[0] + neighbor_message_sum, factor_message[1] + neighbor_message_sum]
            # message.shape = (2,2)
            message = factor_message + neighbor_message_sum

            if sender['label'] > -1:  # verify labeled
                self._trace[sender.index, :] = sender['label']
            else:
                # when the messages are equal, use influence score - hard code
                if message[0, 0] == message[0, 1]:
                    self._trace[sender.index][0] = sender['influence']
                else:
                    self._trace[sender.index][0] = int(message[0, 1] > message[0, 0])

                if message[1, 0] == message[1, 1]:
                    self._trace[sender.index][1] = sender['influence']
                else:
                    self._trace[sender.index][1] = int(message[1, 1] > message[1, 0])

            # self._trace[sender.index].shape = (2,)
            self._messages[sender.index] = message[[0, 1], self._trace[sender.index]]

    def _retrieve_label(self):
        """
        retrieve label from trace
        :return:
        """
        for i in self._component_root:
            self._graph.vs[i]['label'] = self._trace[i, 0]
            for trace_back in self._inference_route:
                if trace_back['label'] < 0:
                    parent_label = self._graph.vs[trace_back['parent']]['label']
                    trace_back['label'] = self._trace[trace_back.index, parent_label]


class GraphicalModelWithLoop(GraphicalModelInference):
    """
    Hard-implement for simple graphical model inference
    has loops, using loopy belief algorithm, binary classification.
    """
    def __init__(self, i_graph):
        """
        :param i_graph: input graph as Graph instance of igraph
        """
        logging.info('GraphicalModelWithLoop __init__')

        GraphicalModelInference.__init__(self, i_graph)
        # self._messages[i] messages on edge i:
        # self._messages[i][0]: message from source -> targer
        # self._messages[i][1]: message from target -> source
        self._messages = np.zeros((self._edges_num, 2, 2))

    def inference(self):
        """
        inference process
        :return:
        """
        logging.info('GraphicalModelWithLoop inference')

        self._send_flooding_message()
        self._retrieve_label()

    def _send_flooding_message(self, threshold=None):
        """
        send messages simultaneously on all edges (both directions)
        :param threshold: number of sending loops
        :return:
        """
        if threshold is None:
            threshold = self._edges_num * 2
            # threshold = self._edges_num // 2

        for _ in range(threshold+1):
            temp_message = np.zeros((self._edges_num, 2, 2))

            # simultaneously sending message on each edge (in both directions)
            for edge_id, edge in enumerate(self._graph.es):
                source, target = edge.tuple

                # update message from source -> target
                neighbor_message_sum = np.zeros(2)
                nes = self._graph.vs[source].neighbors()
                for ne in nes:
                    if ne.index != target:
                        connected_edge_id = self._graph.get_eid(ne.index, source)
                        # looking for the direction ne -> source
                        # connected_edge(source -> target) == ne -> source return 0
                        # connected_edge(source -> target) == source -> ne return 1
                        direction = int(self._graph.es[connected_edge_id].source == source)
                        neighbor_message_sum = \
                            neighbor_message_sum + self._messages[connected_edge_id, direction]

                # exp and log are annulled
                # factor_message.shape = (2,2)
                factor_message = np.array([self.factor_calc(-1, edge['weight']),
                                           self.factor_calc(+1, edge['weight'])])

                # message = [factor_message[0] + neighbor_message_sum, factor_message[1] + neighbor_message_sum]
                # message.shape = (2,2)
                message = factor_message + neighbor_message_sum

                source_label = int(self._graph.vs[source]['label'])
                if source_label > -1:  # verify labeled
                    temp_message[edge_id, 0] = message[:, source_label]
                else:
                    temp_message[edge_id, 0] = np.max(message, axis=1)

                # update message from target -> source
                neighbor_message_sum = np.zeros(2)
                nes = self._graph.vs[target].neighbors()
                for ne in nes:
                    if ne.index != source:
                        connected_edge_id = self._graph.get_eid(ne.index, target)
                        # looking for the direction ne -> target
                        # connected_edge(source -> target) == ne -> target return 0
                        # connected_edge(source -> target) == target -> ne return 1
                        direction = int(self._graph.es[connected_edge_id].source == target)
                        neighbor_message_sum = \
                            neighbor_message_sum + self._messages[connected_edge_id, direction]

                # exp and log are annulled
                # factor_message.shape = (2,2)
                factor_message = np.array([self.factor_calc(-1, edge['weight']),
                                           self.factor_calc(+1, edge['weight'])])

                # message = [factor_message[0] + neighbor_message_sum, factor_message[1] + neighbor_message_sum]
                message = factor_message + neighbor_message_sum

                target_label = int(self._graph.vs[target]['label'])
                if target_label > -1:  # verify labeled
                    temp_message[edge_id, 1] = message[:, target_label]
                else:
                    temp_message[edge_id, 1] = np.max(message, axis=1)

            self._messages = temp_message

    def _retrieve_label(self):
        for vertex in self._graph.vs:
            if vertex['label'] < 0:
                neighbor_message_sum = np.zeros(2)

                nes = vertex.neighbors()
                for ne in nes:
                    connected_edge_id = self._graph.get_eid(ne.index, vertex.index)
                    # looking for the direction ne -> vertex
                    # connected_edge(source -> target) == ne -> vertex return 0
                    # connected_edge(source -> target) == vertex -> ne return 1
                    direction = int(self._graph.es[connected_edge_id].source == vertex.index)
                    neighbor_message_sum = \
                        neighbor_message_sum + self._messages[connected_edge_id][direction]
                if neighbor_message_sum[0] == neighbor_message_sum[1]:
                    vertex['label'] = vertex['influence']
                else:
                    vertex['label'] = np.argmax(neighbor_message_sum)
