"""
    @nghia nh
    Support functions
    ---

    search parameters, export results, ...

"""

import copy

from sklearn.model_selection import StratifiedShuffleSplit

from graph_construction import *
from mincut_inference import *
from graphical_model_inference import *

classification_report_form = {'0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              'micro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
                              'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}}
knn_parameters_form = {'start': 0, 'stop': 0, 'step': 0}  # linear space


def sum_classification_report(x1, x2):
    """
    return sum of 2 classification_report as classification_report_form
    :param x1: classification_report
    :param x2: classification_report
    :return:
    """
    result = copy.deepcopy(classification_report_form)
    for n1, n2, n3 in zip(x1, x2, result):
        for v1, v2, v3 in zip(x1[n1], x2[n2], result[n3]):
            result[n3][v3] = x1[n1][v1] + x2[n2][v2]
    return result


def export_cumulative_result(filename, message, data, cumul_scale=1.0):
    """
     Export cumulative result to file
    :param filename: file to export
    :param message: name of export content
    :param data: classification_report_form
    :param cumul_scale: scale of export result
    :return:
    """
    with open(filename, 'a') as f:
        f.writelines('\n' + message + '\n')
        f.write("\n%15s%10s%10s%10s%10s\n" %
                ("", "precision", "recall", "f1-score", "support"))
        for att in data:
            f.write("%15s" % att)
            for val in data[att]:
                f.write("%10.2f" % round(data[att][val] / cumul_scale, 2))
            f.write('\n')


def error_rate(y_predict, y_true):
    '''
    calc error rate
    :param y_predict: predicted label
    :param y_true: true label
    :return:
    '''
    data_len = len(y_predict)
    y_np_predict = np.array(y_predict).reshape(data_len, 1)
    y_np_true = np.array(y_true).reshape(data_len, 1)
    return np.sum(abs(y_np_predict - y_np_true))/data_len


def search_mincut_knn(x, y, knn_parameters):
    """
    search the best k for mincut approach using knn graph
    :param x: train data
    :param y: train label
    :param knn_parameters: k search range
    :return: k with lowest error rate
    """
    logging.info('search_mincut_knn')

    search_list = np.arange(knn_parameters['start'],
                            knn_parameters['stop']+1,
                            knn_parameters['step'])
    k_best = 0
    error_rate_best = 1
    for k_search in search_list:
        sss_labeled = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for train_index, test_index in sss_labeled.split(x, y):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            graph = GraphConstruction(x_train, y_train, x_test)
            graph.construct_knn_graph(k=k_search)
            mincut = MincutInference(graph.get_graph)
            mincut.inference()

            # calc error rate
            y_test_predict = mincut.get_label[-graph.get_unlabeled_data_number:]
            error = error_rate(y_test_predict, y_test)
            if error < error_rate_best:
                error_rate_best = error
                k_best = k_search

    logging.info('k best = %s, error_rate = %s', k_best, error_rate_best)
    return k_best


def ssl_mincut_knn(x_l, y_l, x_u, knn_paras):
    """
    train mincut approach with knn graph
    :param x_l: labeled data
    :param y_l: labeled label
    :param x_u: unlabeled data
    :param knn_paras: k search range
    :return: predicted label for x_u
    """
    logging.info('Train ssl_mincut_knn')

    train_graph = GraphConstruction(x_l, y_l, x_u)
    k_best = search_mincut_knn(x_l, y_l, knn_paras)
    train_graph.construct_knn_graph(k=k_best)
    mincut = MincutInference(train_graph.get_graph)
    mincut.inference()
    y_mincut_predict = mincut.get_label[-train_graph.get_unlabeled_data_number:]

    # return only unlabeled label
    return y_mincut_predict


def ssl_mincut_mst(x_l, y_l, x_u):
    """
    train mincut approach with minimum spanning tree component graph
    :param x_l: labeled data
    :param y_l: labeled label
    :param x_u: unlabeled data
    :return: predicted label for x_u
    """
    logging.info('ssl_mincut_mst')

    train_graph = GraphConstruction(x_l, y_l, x_u)
    train_graph.construct_mst_graph()
    mincut = MincutInference(train_graph.get_graph)
    mincut.inference()
    y_mincut_predict = mincut.get_label[-train_graph.get_unlabeled_data_number:]

    # return only unlabeled label
    return y_mincut_predict


def ssl_graphical_mst(x_l, y_l, x_u):
    """
    train graphical model with minimum spanning tree component graph
    :param x_l: labeled data
    :param y_l: labeled label
    :param x_u: unlabeled data
    :return: predicted label for x_u
    """
    logging.info('ssl_graphical_mst')

    train_graph = GraphConstruction(x_l, y_l, x_u)
    train_graph.construct_mst_graph()
    graphical = GraphicalModelWithMST(train_graph.get_graph)
    graphical.inference()
    y_graphical_predict = graphical.get_label[-train_graph.get_unlabeled_data_number:]

    # return only unlabeled label
    return y_graphical_predict


def search_mincut_knn_mst(x, y, knn_parameters):
    """
    search the best k for mincut approach using knn graph and minimum spanning tree
    :param x: train data
    :param y: train label
    :param knn_parameters: k search range
    :return: k with lowest error rate
    """
    logging.info('search_mincut_knn_mst')

    search_list = np.arange(knn_parameters['start'],
                            knn_parameters['stop']+1,
                            knn_parameters['step'])
    k_best = 0
    error_rate_best = 1
    for k_search in search_list:
        sss_labeled = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for train_index, test_index in sss_labeled.split(x, y):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            graph = GraphConstruction(x_train, y_train, x_test)
            graph.construct_knn_graph(k=k_search)
            graph.construct_mst_graph()
            mincut = MincutInference(graph.get_graph)
            mincut.inference()

            # calc error rate
            y_test_predict = mincut.get_label[-graph.get_unlabeled_data_number:]
            error = error_rate(y_test_predict, y_test)
            if error < error_rate_best:
                error_rate_best = error
                k_best = k_search

    logging.info('k best = %s, error_rate = %s', k_best, error_rate_best)
    return k_best


def search_graphical_knn_mst(x, y, knn_parameters):
    """
    search the best k for graphical model using knn graph and minimum spanning tree
    :param x: train data
    :param y: train label
    :param knn_parameters: k search range
    :return: k with lowest error rate
    """
    logging.info('search_graphical_knn_mst')

    search_list = np.arange(knn_parameters['start'],
                            knn_parameters['stop']+1,
                            knn_parameters['step'])
    k_best = 0
    error_rate_best = 1
    for k_search in search_list:
        sss_labeled = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
        for train_index, test_index in sss_labeled.split(x, y):
            x_train = x[train_index]
            y_train = y[train_index]
            x_test = x[test_index]
            y_test = y[test_index]
            graph = GraphConstruction(x_train, y_train, x_test)
            graph.construct_knn_graph(k=k_search)
            graph.construct_mst_graph()
            graphical = GraphicalModelWithMST(graph.get_graph)
            graphical.inference()

            # calc error rate
            y_test_predict = graphical.get_label[-graph.get_unlabeled_data_number:]
            error = error_rate(y_test_predict, y_test)
            if error < error_rate_best:
                error_rate_best = error
                k_best = k_search

    logging.info('k best = %s, error_rate = %s', k_best, error_rate_best)
    return k_best


def ssl_mincut_knn_mst(x_l, y_l, x_u, knn_paras):
    """
    train mincut approach with knn graph and minimum spanning tree
    :param x_l: labeled data
    :param y_l: labeled label
    :param x_u: unlabeled data
    :param knn_paras: k search range
    :return: predicted label for x_u
    """
    logging.info('ssl_mincut_knn_mst')

    train_graph = GraphConstruction(x_l, y_l, x_u)
    k_best = search_mincut_knn_mst(x_l, y_l, knn_paras)
    train_graph.construct_knn_graph(k=k_best)
    train_graph.construct_mst_graph()
    mincut_knn_mst = MincutInference(train_graph.get_graph)
    mincut_knn_mst.inference()
    y_mincut_predict = mincut_knn_mst.get_label[-train_graph.get_unlabeled_data_number:]

    # return only unlabeled label
    return y_mincut_predict


def ssl_graphical_knn_mst(x_l, y_l, x_u, knn_paras):
    """
    train graphical model with knn graph and minimum spanning tree
    :param x_l: labeled data
    :param y_l: labeled label
    :param x_u: unlabeled data
    :param knn_paras: k search range
    :return: predicted label for x_u
    """
    logging.info('ssl_graphical_knn_mst')

    train_graph = GraphConstruction(x_l, y_l, x_u)
    k_best = search_graphical_knn_mst(x_l, y_l, knn_paras)
    train_graph.construct_knn_graph(k=k_best)
    train_graph.construct_mst_graph()
    graphical_knn_mst = GraphicalModelWithMST(train_graph.get_graph)
    graphical_knn_mst.inference()
    y_graphical_predict = graphical_knn_mst.get_label[-train_graph.get_unlabeled_data_number:]

    # return only unlabeled label
    return y_graphical_predict


# TODO Finish unittest
class TestUtility(unittest.TestCase):
    def test_error_rate(self):
        y_true = [1,1,1,1]
        y_predict = [0,1,1,1]
        self.assertEqual(0.25, error_rate(y_predict, y_true))