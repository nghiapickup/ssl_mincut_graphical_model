"""
    @nghia nh
    ---
    Test cases

"""
import sys
import copy
import logging

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from data.abalone_processing import AbaloneData
from data.mushroom_processing import MushroomData
from data.anuran_calls_processing import AnuranCallData
from data.newsgroups_processing import NewsgroupsData
from data.digit_processing import DigitData
from data.reuters_processing import ReutersData
from graph_construction import GraphConstruction
from mincut_inference import MincutInference, RandomizeMincutInference
from graphical_model_inference import GraphicalModelWithTree, GraphicalModelWithLoop
from utility import ResultExporter, ParamSearch


# log config
LOG_FILE = 'source.log'

logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] "
    "[%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(LOG_FILE)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

# consoleHandler = logging.StreamHandler()
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)


###############################################################################
# Data dir

# Where all data are located, should only one location
data_folder_dir = 'data/'

# Where each data are located in data_folder
data_dir = {
    'abalone': data_folder_dir + AbaloneData.default_file_dir,
    'mushroom': data_folder_dir + MushroomData.default_file_dir,
    'anuran': data_folder_dir + AnuranCallData.default_file_dir,
    '20news': data_folder_dir + NewsgroupsData.default_folder,
    'reuters': data_folder_dir + ReutersData.default_folder
}


###############################################################################
# transformer and classifier interface

class GraphTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, graph_type='knn', metric='euclidean'):
        self.graph_type = graph_type
        self.metric = metric
        self.transformer = None

    def fit(self, x, y_l, **kwargs_graph):
        self.transformer = GraphConstruction(x_l=x[0], y_l=y_l,
                                             x_u=x[1], metric=self.metric)
        self.transformer.construct_graph(graph_type=self.graph_type, **kwargs_graph)

        return self

    def transform(self, x=None):
        return self.transformer


class GraphInference(BaseEstimator, ClassifierMixin):
    def __init__(self, model_type='mincut'):
        self.model_map = {
            'mincut': MincutInference,
            'mincut_randomize': RandomizeMincutInference,
            'graphical_tree': GraphicalModelWithTree,
            'graphical_loop': GraphicalModelWithLoop
        }
        self.model_type = model_type

        self.inference = None
        self.unlabeled_number = None

    def fit(self, graph_constructor, y=None):
        self.unlabeled_number = graph_constructor.x_u_number

        try:
            self.inference = self.model_map[self.model_type](graph_constructor.graph)
        except KeyError:
            logging.error('GraphInference: Non supported model %s.' % self.model_type)
            raise

        self.inference.inference()

        return self

    def transform(self, x=None):
        pass

    def predict(self, x=None, y=None):
        return self.inference.labels[-self.unlabeled_number:]


###############################################################################
# Experiment setup

class ExperimentSetUp:
    def __init__(self, data_processor, result_filename):
        self.cv_fold_number = 5
        self.knn_param = {'start': 1, 'stop': 10, 'step': 1}
        self.data_processor = data_processor.transform()
        self.result_filename = result_filename

        # SSL setup
        self.mincut_param = {
            'infer__model_type': ['mincut'],
            'trans__metric': ['euclidean', 'gaussian', 'cosine'],
            'trans__graph_type': ['knn', 'mst', 'knn_mst']
        }
        self.mincut_randomize_param = {
            'infer__model_type': ['mincut_randomize'],
            'trans__metric': ['euclidean', 'gaussian', 'cosine'],
            'trans__graph_type': ['knn', 'mst', 'knn_mst']
        }
        self.graphical_tree_param = {
            'infer__model_type': ['graphical_tree'],
            'trans__metric': ['euclidean', 'gaussian', 'cosine'],
            'trans__graph_type': ['mst', 'knn_mst']
        }
        self.graphical_loop_param = {
            'infer__model_type': ['graphical_loop'],
            'trans__metric': ['euclidean', 'gaussian', 'cosine'],
            'trans__graph_type': ['knn', 'mst', 'knn_mst']
        }

        self.ssl_param_grid = ParameterGrid([
            self.mincut_param,
            self.mincut_randomize_param,
            self.graphical_tree_param,
            self.graphical_loop_param])

        # pipeline
        self.ssl_experiment_pipeline = Pipeline([
            ('trans', GraphTransformer()),
            ('infer', GraphInference())])

        # SL setup
        self.sl_models = {
            'svm': SVC(gamma='scale'),
            'logreg': LogisticRegression()
        }

    def process_ssl(self, unlabeled_size=0.7):
        """
        Experiment on ssl data only.
        The input data is split into train-test sets.
        The ssl methods are train instances of all input
        and predict labeled of test set.
        :param unlabeled_size: size of unlabeled data
        :return:
        """

        # cumulative results
        result = []
        for param_set in self.ssl_param_grid:
            result.append(ResultExporter(
                param_set['infer__model_type']
                + '_'
                + param_set['trans__metric']
                + '_'
                + param_set['trans__graph_type'])
            )

        # split labeled and unlabeled set
        sss_data = StratifiedShuffleSplit(n_splits=self.cv_fold_number,
                                          test_size=unlabeled_size,
                                          random_state=0)
        for train_index, test_index in sss_data.split(np.zeros(self.data_processor.x_number),
                                                      self.data_processor.y):
            x_l, y_l, x_u, y_u = self.data_processor.extract(train_index, test_index)

            for test_id, param_set in enumerate(self.ssl_param_grid):

                # set basic params first
                self.ssl_experiment_pipeline.set_params(**param_set)

                # cv search parameter if necessary

                # graph_param: search k for KNN graph on labeled set
                k=0
                if 'knn' in param_set['trans__graph_type']:
                    k = ParamSearch.knn_graph_search(
                        x_l, y_l,
                        self.ssl_experiment_pipeline,
                        self.knn_param)

                y_u_predict = self.ssl_experiment_pipeline.fit(
                    [x_l, x_u], y_l, trans__k=k).predict(None)

                # only sum score on y_u
                result[test_id].sum_report(classification_report(y_u, y_u_predict, output_dict=True))

        # export result
        for test_case in result:
            test_case.export(self.result_filename, scale=self.cv_fold_number)

    def process_sl(self, test_size=0.3, unlabeled_size=0.7):

        # cumulative results
        result = {}
        sl_model_names = self.sl_models.keys()

        for param_set in self.ssl_param_grid:
            ssl_name = param_set['infer__model_type'] \
                       + '_' \
                       + param_set['trans__metric'] \
                       + '_' \
                       + param_set['trans__graph_type']
            for sl_name in sl_model_names:
                model_name = sl_name + '_' + ssl_name
                result[model_name] = ResultExporter(model_name)

        # split labeled and unlabeled set
        sl_sss_data = StratifiedShuffleSplit(n_splits=self.cv_fold_number,
                                          test_size=test_size,
                                          random_state=0)
        for train_index, test_index in sl_sss_data.split(np.zeros(self.data_processor.x_number),
                                                      self.data_processor.y):
            x_train, y_train, x_test, y_test = self.data_processor.extract(train_index, test_index)

            ssl_sss_data = StratifiedShuffleSplit(n_splits=self.cv_fold_number,
                                              test_size=unlabeled_size,
                                              random_state=0)
            for labeled_index, unlabeled_index in ssl_sss_data.split(x_train, y_train):
                x_l, y_l = x_train[labeled_index], y_train[labeled_index]
                x_u, _ = x_train[unlabeled_index], y_train[unlabeled_index]
                y_train_predict = copy.deepcopy(y_train)

                for param_set in self.ssl_param_grid:

                    # set basic params first
                    self.ssl_experiment_pipeline.set_params(**param_set)

                    # cv search parameter if necessary

                    # graph_param: search k for KNN graph on labeled set
                    k = 0
                    if 'knn' in param_set['trans__graph_type']:
                        k = ParamSearch.knn_graph_search(
                            x_l, y_l,
                            self.ssl_experiment_pipeline,
                            self.knn_param)

                    y_u_predict = self.ssl_experiment_pipeline.fit(
                        [x_l, x_u], y_l, trans__k=k).predict(None)

                    # use the predicted unlabeled data to train test data
                    ssl_name = param_set['infer__model_type'] \
                               + '_' \
                               + param_set['trans__metric'] \
                               + '_' \
                               + param_set['trans__graph_type']
                    y_train_predict[unlabeled_index] = y_u_predict

                    for sl_name in sl_model_names:
                        sl_model = self.sl_models.get(sl_name)
                        sl_model.fit(x_train, y_train_predict)

                        # sum score on y_test
                        report = classification_report(y_test, sl_model.predict(x_test), output_dict=True)
                        result[sl_name + '_' + ssl_name].sum_report(report)

        # export result
        for test_case in result.values():
            test_case.export(self.result_filename, scale=self.cv_fold_number*self.cv_fold_number)


###############################################################################
# Test cases

class Experiment:
    def __init__(self):
        self.experiments_map = {
            'anuran': self.anuran_experiment,
            'mushroom': self.mushroom_experiment,
            'abalone': self.abalone_experiment,
            'newsgroups': self.newsgroups_experiment,
            'digit': self.digit_experiment,
            'reuters': self.reuters_experiment
        }

        self.params_map = {
            'anuran': {'file_dir': data_dir['anuran']},
            'mushroom': {'file_dir': data_dir['mushroom']},
            'abalone': {
                'file_dir': data_dir['abalone'],
                'positive_rings': [5, 6, 7, 8, 9]
            },
            'newsgroups': {
                'folder_dir': data_dir['20news'],
                'categories': [
                    'comp.graphics', 'comp.os.ms-windows.misc',
                    'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'
                ],
                'positive_labels': [
                    'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey'
                ],
                'feature_score': 'tfidf',
                'feature_number': 400,
                'normalize': 'tfidf',
                'scale': 10000.
            },
            'digit': {},
            'reuters': {
                'folder_dir': data_dir['reuters'],
                'positive_labels': 'acq',
                'feature_score': 'tfidf',
                'feature_number': 400,
                'normalize': 'tfidf',
                'scale': 10000.
            }
        }

    def get_experiment(self, exp_name):
        try:
            return self.experiments_map[exp_name](**self.params_map[exp_name])
        except KeyError:
            logging.error('Experiment: Non recognized experiment %s.' % exp_name)
            raise

    @staticmethod
    def abalone_experiment(file_dir, positive_rings):
        """
        Experiment on Abalone data, binary classifier

        Class distribution(training set)
        5	115
        6	259
        7	391
        8	568
        9	689
        10	634
        11	487
        12	267
        13	203
        14	126
        15	103

        :param file_dir: data file location
        :param positive_rings: Value or list of values for positive label
        :return:
        """
        logging.info('Start Abalone Experiment')

        result_file_name = 'abalone_experiment.out'
        data_processor = AbaloneData(file_dir, positive_labels=positive_rings)
        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def mushroom_experiment(file_dir):
        """
        Experiment on Mushroom data, binary classifier

        Class Distribution:
        edible:     4208 (51.8 %)
        poisonous:  3916 (48.2 %)
        total:      8124 instances
        :param file_dir: data file location
        :return:
        """
        logging.info('Start Mushroom Experiment')

        result_file_name = 'mushroom_experiment.out'
        data_processor = MushroomData(file_dir)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def anuran_experiment(file_dir):
        """
        Experiment on Anuran Calls (MFCCs) data, binary classifier
        with positive label Family=Leptodactylidae
        :param file_dir: data file location
        :return:
        """
        logging.info('Start Anuran Experiment')

        result_file_name = 'anuran_experiment.out'
        data_processor = AnuranCallData(file_dir)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def newsgroups_experiment(folder_dir, categories=None, positive_labels=None,
                              feature_score=None, feature_number=600,
                              normalize=None, **kwargs_normalize):
        """
        Experiment on 20Newsgroup data, binary classifier

        Class names
        'alt.atheism',
        'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
        'comp.sys.mac.hardware', 'comp.windows.x',
        'misc.forsale',
        'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
        'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space',
        'soc.religion.christian',
        'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc'

        :param folder_dir: data folder
        :param categories: picked categorizes
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        :return:
        """
        logging.info('Start 20Newsgroup Experiment')

        result_file_name = 'newsgroups_experiment.out'
        data_processor = NewsgroupsData(
            folder_dir=folder_dir, categories=categories, positive_labels=positive_labels,
            feature_score=feature_score, feature_number=feature_number,
            normalize=normalize, **kwargs_normalize)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def digit_experiment():
        """
        Experiment on digit data, binary classifier
        task on Odd vs Even digit.
        The data is loaded directly from sklearn.

        Classes             10
        Samples per class   ~180
        Samples total       1797
        Dimensionality      64
        Features            integers 0-16
        :return: result export to file result_file_name
        """
        logging.info('Start Digit Experiment')

        result_file_name = 'digit_experiment.out'
        data_processor = DigitData()

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp

    @staticmethod
    def reuters_experiment(folder_dir, positive_labels=None,
                              feature_score=None, feature_number=600,
                              normalize=None, **kwargs_normalize):
        """
        Experiment on Reuters 21578 data, binary classifier.
        Number of instances: 19716

        :param folder_dir: data folder
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        :return:
        """
        logging.info('Start Reuters Experiment')

        result_file_name = 'reuters_experiment.out'
        data_processor = ReutersData(
            folder_dir=folder_dir, positive_labels=positive_labels,
            feature_score=feature_score, feature_number=feature_number,
            normalize=normalize, **kwargs_normalize)

        exp = ExperimentSetUp(data_processor, result_file_name)

        return exp


###############################################################################
def main():
    logging.info('Start main()')
    try:
        exp = Experiment()

        exp1 = exp.get_experiment('abalone')
        # exp1 = exp.get_experiment('anuran')
        # exp1 = exp.get_experiment('digit')
        # exp1 = exp.get_experiment('mushroom')
        # exp1 = exp.get_experiment('reuters')
        # exp1 = exp.get_experiment('newsgroups')

        exp1.process_ssl(unlabeled_size=0.7)
        exp1.process_sl(test_size=0.3, unlabeled_size=0.7)

    except BaseException:
        logging.exception('Main exception')
        raise
    finally:
        return 'Done main()'


if __name__ == '__main__':
    status = main()
    sys.exit(status)
