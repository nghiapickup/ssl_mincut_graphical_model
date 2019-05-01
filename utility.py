"""
    @nghia nh
    ---
    Support functions

    search parameters, export results, ...
"""

import copy
import logging

from sklearn.model_selection import StratifiedShuffleSplit


class ResultExporter:
    __REPORT_FORM = {
        '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        'micro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
        'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    }

    def __init__(self, name):
        self.report = ResultExporter.report_form()
        self.name = name

    @staticmethod
    def report_form():
        return copy.deepcopy(ResultExporter.__REPORT_FORM)

    def sum_report(self, x):
        """
        sum in self.report a classification_report
        :param x: classification_report
        :return:
        """
        for n1, n2 in zip(self.report, x):
            for v1, v2 in zip(self.report[n1], x[n2]):
                self.report[n1][v1] = self.report[n1][v1] + x[n2][v2]

    def export(self, filename, scale=1.0, message=None):
        """
         Export self.report to file
        :param filename: file to export
        :param scale: scale of export result
        :param message: name of export content
        :return:
        """
        # TODO verify datatype of input parameters
        with open(filename, 'a') as f:
            if message is not None:
                f.writelines('\n' + message)
            f.writelines('\n' + self.name + '\n')
            f.write("\n%15s%10s%10s%10s%10s\n" %
                    ("", "precision", "recall", "f1-score", "support"))
            for att in self.report:
                f.write("%15s" % att)
                for val in self.report[att]:
                    f.write("%10.2f" % round(self.report[att][val] / scale, 2))
                f.write('\n')


###############################################################################
class ParamSearch:
    @staticmethod
    def knn_graph_search(x, y, pipeline, knn_param):
        """
        search the best k for KNN graph
        :param x: train data
        :param y: train label
        :param pipeline: learning pipeline
        :param knn_param: k search range
        :return: k with best score
        """
        logging.info('knn_graph_search')

        search_list = range(knn_param['start'], knn_param['stop']+1, knn_param['step'])
        k_best = knn_param['start']
        best_score = 0
        for k_search in search_list:
            sss_labeled = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=0)
            for train_index, test_index in sss_labeled.split(x, y):
                x_train = x[train_index]
                y_train = y[train_index]
                x_test = x[test_index]
                y_test = y[test_index]

                pipeline.fit([x_train, x_test], y_train, trans__k = k_search)
                # Pipeline's score directly call ClassifierMixin's score
                # and ClassifierMixin's score will compare y_true input vs its predict result
                score = pipeline.score(None, y_test)

                if score > best_score:
                    best_score = score
                    k_best = k_search

        return k_best
