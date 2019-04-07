"""
    @nghia nh
    ---
    Test cases

"""
import logging

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit

from data.abalone_processing import AbaloneData
from data.mushroom_processing import MushroomData
from data.anuran_calls_processing import AnuranCallData
from data.newsgroups_processing import NewsgroupsData
from graph_construction import GraphConstruction
from mincut_inference import MincutInference
from graphical_model_inference import GraphicalModelWithTree
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
            'graphical_tree': GraphicalModelWithTree
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

class Experiment():
    def __init__(self, data_processor, result_filename):
        self.cv_fold_number = 5
        self.knn_param = {'start': 1, 'stop': 10, 'step': 1}

        self.mincut_param = {
            'infer__model_type': ['mincut'],
            'trans__metric': ['euclidean', 'rbf', 'cosine'],
            'trans__graph_type': ['knn', 'mst', 'knn_mst']}
        self.graphical_tree_param = {
            'infer__model_type': ['graphical_tree'],
            'trans__metric': ['euclidean', 'rbf', 'cosine'],
            'trans__graph_type': ['mst', 'knn_mst']}

        self.param_grid = ParameterGrid([self.mincut_param, self.graphical_tree_param])

        # setup pipeline
        self.experiment_pipeline = Pipeline([
            ('trans', GraphTransformer()),
            ('infer', GraphInference())])

        self.data_processor = data_processor.transform()
        self.result_filename = result_filename

    def process_ssl(self, unlabeled_size=0.5):

        # cumulative results
        result = []
        for param_set in self.param_grid:
            result.append(ResultExporter(
                param_set['infer__model_type'] + '_' +\
                param_set['trans__metric'] + '_' + \
                param_set['trans__graph_type'])
            )

        # split labeled and unlabeled set
        sss_data = StratifiedShuffleSplit(n_splits=self.cv_fold_number,
                                          test_size=unlabeled_size,
                                          random_state=0)
        for train_index, test_index in sss_data.split(np.zeros(self.data_processor.x_number), self.data_processor.y):
            x_l, y_l, x_u, y_u = self.data_processor.extract(train_index, test_index)

            for test_id, param_set in enumerate(self.param_grid):

                # set basic params first
                self.experiment_pipeline.set_params(**param_set)

                # cv search parameter if necessary

                # graph_param: search k for KNN graph on labeled set
                k=0
                if 'knn' in param_set['trans__graph_type']:
                    k = ParamSearch.knn_graph_search(
                        x_l, y_l,
                        self.experiment_pipeline,
                        self.knn_param)

                y_u_predict = self.experiment_pipeline.fit(
                    [x_l, x_u], y_l, trans__k=k).predict(None)

                # only sum score on y_u
                result[test_id].sum_report(classification_report(y_u, y_u_predict, output_dict=True))

        # export result
        for test_case in result:
            test_case.export(self.result_filename, scale=self.cv_fold_number)


###############################################################################
# Test case

data_folder_dir = {
    'abalone': 'data/abalone/',
    'mushroom': 'data/mushroom/',
    'anuran': 'data/anuran_calls/',
    '20news': 'data/20newsbydate/'
}

data_dir = {
    'abalone': data_folder_dir['abalone'] + AbaloneData.default_filename,
    'mushroom': data_folder_dir['mushroom'] + MushroomData.default_filename,
    'anuran': data_folder_dir['anuran'] + AnuranCallData.default_filename,
    '20news': data_folder_dir['20news'] + NewsgroupsData.default_file_folder
}


def abalone_experiment(file_dir, positive_rings, unlabeled_size=0.5):
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
    :param unlabeled_size: size of unlabeled data
    :return: result export to file result_file_name
    """
    logging.info('Start Abalone Experiment')

    result_file_name = 'abalone_experiment.out'
    data_processor = AbaloneData(file_dir, positive_labels=positive_rings)

    exp = Experiment(data_processor, result_file_name)
    exp.process_ssl(unlabeled_size=unlabeled_size)


def mushroom_experiment(file_dir, unlabeled_size=0.5):
    """
    Experiment on Mushroom data, binary classifier

    Class Distribution:
    edible:     4208 (51.8 %)
    poisonous:  3916 (48.2 %)
    total:      8124 instances
    :param file_dir: data file location
    :param unlabeled_size: size of unlabeled data
    :return: result export to file result_file_name
    """
    logging.info('Start Mushroom Experiment')

    result_file_name = 'mushroom_experiment.out'
    data_processor = MushroomData(file_dir)

    exp = Experiment(data_processor, result_file_name)
    exp.process_ssl(unlabeled_size=unlabeled_size)


def anuran_experiment(file_dir, unlabeled_size=0.5):
    """
    Experiment on Anuran Calls (MFCCs) data, binary classifier
    with positive label Family=Leptodactylidae
    :param file_dir: data file location
    :param unlabeled_size: size of unlabeled data
    :return: result export to file result_file_name
    """
    logging.info('Start Anuran Experiment')

    result_file_name = 'anuran_experiment.out'
    data_processor = AnuranCallData(file_dir)

    exp = Experiment(data_processor, result_file_name)
    exp.process_ssl(unlabeled_size=unlabeled_size)


def newsgroups_experiment(folder_dir, unlabeled_size=0.5, categories=None, positive_labels=None,
                          feature_score=None, feature_number=600, normalize=None, **kwargs_normalize):
    """
    Experiment on 20Newsgroup data, binary classifier

    Class names
    'alt.atheism',
    'comp.graphics',
    'comp.os.ms-windows.misc',
    'comp.sys.ibm.pc.hardware',
    'comp.sys.mac.hardware',
    'comp.windows.x',
    'misc.forsale',
    'rec.autos',
    'rec.motorcycles',
    'rec.sport.baseball',
    'rec.sport.hockey',
    'sci.crypt',
    'sci.electronics',
    'sci.med',
    'sci.space',
    'soc.religion.christian',
    'talk.politics.guns',
    'talk.politics.mideast',
    'talk.politics.misc',
    'talk.religion.misc'
    :param folder_dir: data folder
    :param unlabeled_size: size of unlabeled data
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

    exp = Experiment(data_processor, result_file_name)
    exp.process_ssl(unlabeled_size=unlabeled_size)


###############################################################################
def main():
    logging.info('Start main()')
    try:
        abalone_experiment(
            file_dir=data_dir['abalone'],
            positive_rings=[5, 6, 7, 8, 9],
            unlabeled_size=0.7)

        anuran_experiment(file_dir=data_dir['anuran'], unlabeled_size=0.7)

        mushroom_experiment(file_dir=data_dir['mushroom'], unlabeled_size=0.7)

        newsgroups_experiment(
            folder_dir=data_dir['20news'], unlabeled_size=0.7,
            categories=[
                'comp.graphics', 'comp.os.ms-windows.misc',
                'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x',

                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball','rec.sport.hockey',
                ],
            positive_labels=[
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey',
            ],
            feature_score='tfidf', feature_number=600,
            normalize='tfidf', scale = 10000.
        )
    except BaseException:
        logging.exception('Main exception')
        raise


if __name__ == '__main__':
    main()
