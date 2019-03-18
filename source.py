"""
    @nghia nh
    ===
"""

from sklearn.metrics import classification_report

from data.abalone_processing import AbaloneData
from data.mushroom_processing import MushroomData
from data.anuran_calls_processing import AnuranCallData
from utility import *

# log config
LOG_FILE = 'source.log'
logging.basicConfig()

logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)

fileHandler = logging.FileHandler(LOG_FILE)
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


def abalone_experiment(positive_rings, unlabeled_size=0.5):
    """
    Experiment on Abalone dataset, only binary classifier
    :param positive_rings: Value or list of values for positive label
    :param unlabeled_size: size of unlabeled data
    :return: result export to file result_file_name
    """
    logging.info('Start Abalone Experiment')
    data = AbaloneData('data/abalone/abalone.data.source.test.txt')  # test
    # data = AbaloneData('data/abalone/' + AbaloneData.default_file_name)

    # Class distribution(training set)
    #   5	115
    # 	6	259
    # 	7	391
    # 	8	568
    # 	9	689
    # 	10	634
    # 	11	487
    # 	12	267
    # 	13	203
    # 	14	126
    # 	15	103

    x, y = data.get_binary_class_data(positive_rings)

    # spit with cv fold = 5
    cv_fold_number = 5

    # cumulative results
    result_file_name = 'abalone_experiment.out'
    ssl_mincut_knn_result = copy.deepcopy(classification_report_form)
    ssl_graphical_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_knn_mst_result = copy.deepcopy(classification_report_form)
    ssl_graphical_knn_mst_result = copy.deepcopy(classification_report_form)

    # split labeled and unlabeled set
    sss_data = StratifiedShuffleSplit(n_splits=cv_fold_number, test_size=unlabeled_size, random_state=0)
    for train_index, test_index in sss_data.split(x, y):
        x_l = x[train_index]
        y_l = y[train_index]
        x_u = x[test_index]
        y_u = y[test_index]
        knn_paras = copy.deepcopy(knn_parameters_form)
        knn_paras['start'] = 1
        knn_paras['stop'] = 15
        knn_paras['step'] = 1

        # (method 1) SSL mincut using KNN graph
        y_u_predict_mincut_knn = ssl_mincut_knn(x_l, y_l, x_u, knn_paras)

        # (method 2 & 3)
        # SSL graphical model and mincut using minimum spanning tree graph
        # 2 methods may use 2 different trees
        y_u_predict_mincut_mst = ssl_mincut_mst(x_l, y_l, x_u)
        y_u_predict_graphical_mst = ssl_graphical_mst(x_l, y_l, x_u)

        # (method 4 & 5)
        # SSL graphical model and mincut using minimum spanning trees on KNN graph's components
        y_u_predict_mincut_knn_mst = ssl_mincut_knn_mst(x_l, y_l, x_u, knn_paras)
        y_u_predict_graphical_knn_mst = ssl_graphical_knn_mst(x_l, y_l, x_u, knn_paras)

        ssl_mincut_knn_result = sum_classification_report(
            ssl_mincut_knn_result,
            classification_report(y_u, y_u_predict_mincut_knn, output_dict=True))
        ssl_graphical_mst_result = sum_classification_report(
            ssl_graphical_mst_result,
            classification_report(y_u, y_u_predict_graphical_mst, output_dict=True))
        ssl_mincut_mst_result = sum_classification_report(
            ssl_mincut_mst_result,
            classification_report(y_u, y_u_predict_mincut_mst, output_dict=True))
        ssl_mincut_knn_mst_result = sum_classification_report(
            ssl_mincut_knn_mst_result,
            classification_report(y_u, y_u_predict_mincut_knn_mst, output_dict=True))
        ssl_graphical_knn_mst_result = sum_classification_report(
            ssl_graphical_knn_mst_result,
            classification_report(y_u, y_u_predict_graphical_knn_mst, output_dict=True))

    # export result
    export_cumulative_result(result_file_name, "ssl_mincut_knn_result",
                             ssl_mincut_knn_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_mst_result",
                             ssl_graphical_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_mst_result",
                             ssl_mincut_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_knn_mst_result",
                             ssl_mincut_knn_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_knn_mst_result",
                             ssl_graphical_knn_mst_result, cumul_scale=cv_fold_number)


def mushroom_experiment(unlabeled_size=0.5):

    logging.info('Start Mushroom Experiment')
    data = MushroomData('data/mushroom/agaricus-lepiota.data.source.test.txt')  # test
    # data = MushroomData('data/mushroom/' + MushroomData.default_file_name)

    # Class Distribution:
    # --    edible: 4208(51.8 %)
    # -- poisonous: 3916(48.2 %)
    # --     total: 8124 instances
    x, y = data.get_basic_data()

    # spit with cv fold = 5
    cv_fold_number = 5

    # cumulative results
    result_file_name = 'mushroom_experiment.out'
    ssl_mincut_knn_result = copy.deepcopy(classification_report_form)
    ssl_graphical_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_knn_mst_result = copy.deepcopy(classification_report_form)
    ssl_graphical_knn_mst_result = copy.deepcopy(classification_report_form)

    # split labeled and unlabeled set
    sss_data = StratifiedShuffleSplit(n_splits=cv_fold_number, test_size=unlabeled_size, random_state=0)
    for train_index, test_index in sss_data.split(x, y):
        # convert data to scalar after getting train
        # THIS mus be processing only on train data
        # and apply same setting for test data
        x_l, x_u = data.get_scalar_data(train_index, test_index)
        y_l = y[train_index]
        y_u = y[test_index]

        # (method 1) SSL mincut using KNN graph
        knn_paras = copy.deepcopy(knn_parameters_form)
        knn_paras['start'] = 1
        knn_paras['stop'] = 15
        knn_paras['step'] = 1
        y_u_predict_mincut_knn = ssl_mincut_knn(x_l, y_l, x_u, knn_paras)

        # (method 2 & 3)
        # SSL graphical model and mincut using minimum spanning tree graph
        # 2 methods may use 2 different trees
        y_u_predict_mincut_mst = ssl_mincut_mst(x_l, y_l, x_u)
        y_u_predict_graphical_mst = ssl_graphical_mst(x_l, y_l, x_u)

        # (method 4 & 5)
        # SSL graphical model and mincut using minimum spanning trees on KNN graph's components
        y_u_predict_mincut_knn_mst = ssl_mincut_knn_mst(x_l, y_l, x_u, knn_paras)
        y_u_predict_graphical_knn_mst = ssl_graphical_knn_mst(x_l, y_l, x_u, knn_paras)

        ssl_mincut_knn_result = sum_classification_report(
            ssl_mincut_knn_result,
            classification_report(y_u, y_u_predict_mincut_knn, output_dict=True))
        ssl_graphical_mst_result = sum_classification_report(
            ssl_graphical_mst_result,
            classification_report(y_u, y_u_predict_graphical_mst, output_dict=True))
        ssl_mincut_mst_result = sum_classification_report(
            ssl_mincut_mst_result,
            classification_report(y_u, y_u_predict_mincut_mst, output_dict=True))
        ssl_mincut_knn_mst_result = sum_classification_report(
            ssl_mincut_knn_mst_result,
            classification_report(y_u, y_u_predict_mincut_knn_mst, output_dict=True))
        ssl_graphical_knn_mst_result = sum_classification_report(
            ssl_graphical_knn_mst_result,
            classification_report(y_u, y_u_predict_graphical_knn_mst, output_dict=True))

        # export result
    export_cumulative_result(result_file_name, "ssl_mincut_knn_result",
                             ssl_mincut_knn_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_mst_result",
                             ssl_graphical_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_mst_result",
                             ssl_mincut_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_knn_mst_result",
                             ssl_mincut_knn_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_knn_mst_result",
                             ssl_graphical_knn_mst_result, cumul_scale=cv_fold_number)


def anuran_experiment(unlabeled_size=0.5):
    """
    Experiment using Anuran calls dataset, only binary on Family=Leptodactylidae
    """
    logging.info('Start Experiment Anuran Calls (MFCCs) Data')
    data = AnuranCallData('data/anuran_calls/Frogs_MFCCs.source.test.csv')  # test
    # data = AnuranCallData('data/anuran_calls/' + AnuranCallData.default_file_name)

    # Class distribution(training set)
    # #instances 7195
    # #features 22

    x, y = data.get_binary_class_data()

    # spit with cv fold = 5
    cv_fold_number = 5

    # cumulative results
    result_file_name = 'anuran_experiment.out'
    ssl_mincut_knn_result = copy.deepcopy(classification_report_form)
    ssl_graphical_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_mst_result = copy.deepcopy(classification_report_form)
    ssl_mincut_knn_mst_result = copy.deepcopy(classification_report_form)
    ssl_graphical_knn_mst_result = copy.deepcopy(classification_report_form)

    # split labeled and unlabeled set
    sss_data = StratifiedShuffleSplit(n_splits=cv_fold_number, test_size=unlabeled_size, random_state=0)
    for train_index, test_index in sss_data.split(x, y):
        x_l = x[train_index]
        y_l = y[train_index]
        x_u = x[test_index]
        y_u = y[test_index]

        # (method 1) SSL mincut using KNN graph
        knn_paras = copy.deepcopy(knn_parameters_form)
        knn_paras['start'] = 1
        knn_paras['stop'] = 15
        knn_paras['step'] = 1
        y_u_predict_mincut_knn = ssl_mincut_knn(x_l, y_l, x_u, knn_paras)

        # (method 2 & 3)
        # SSL graphical model and mincut using minimum spanning tree graph
        # 2 methods may use 2 different trees
        y_u_predict_mincut_mst = ssl_mincut_mst(x_l, y_l, x_u)
        y_u_predict_graphical_mst = ssl_graphical_mst(x_l, y_l, x_u)

        # (method 4 & 5)
        # SSL graphical model and mincut using minimum spanning trees on KNN graph's components
        y_u_predict_mincut_knn_mst = ssl_mincut_knn_mst(x_l, y_l, x_u, knn_paras)
        y_u_predict_graphical_knn_mst = ssl_graphical_knn_mst(x_l, y_l, x_u, knn_paras)

        ssl_mincut_knn_result = sum_classification_report(
            ssl_mincut_knn_result,
            classification_report(y_u, y_u_predict_mincut_knn, output_dict=True))
        ssl_graphical_mst_result = sum_classification_report(
            ssl_graphical_mst_result,
            classification_report(y_u, y_u_predict_graphical_mst, output_dict=True))
        ssl_mincut_mst_result = sum_classification_report(
            ssl_mincut_mst_result,
            classification_report(y_u, y_u_predict_mincut_mst, output_dict=True))
        ssl_mincut_knn_mst_result = sum_classification_report(
            ssl_mincut_knn_mst_result,
            classification_report(y_u, y_u_predict_mincut_knn_mst, output_dict=True))
        ssl_graphical_knn_mst_result = sum_classification_report(
            ssl_graphical_knn_mst_result,
            classification_report(y_u, y_u_predict_graphical_knn_mst, output_dict=True))

        # export result
    export_cumulative_result(result_file_name, "ssl_mincut_knn_result",
                             ssl_mincut_knn_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_mst_result",
                             ssl_graphical_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_mst_result",
                             ssl_mincut_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_mincut_knn_mst_result",
                             ssl_mincut_knn_mst_result, cumul_scale=cv_fold_number)
    export_cumulative_result(result_file_name, "ssl_graphical_knn_mst_result",
                             ssl_graphical_knn_mst_result, cumul_scale=cv_fold_number)


def main():
    logging.info('Start main()')
    try:
        # abalone_experiment(positive_rings=[5, 6, 7, 8, 9], unlabeled_size=0.5)
        # mushroom_experiment(unlabeled_size=0.5)
        anuran_experiment(unlabeled_size=0.5)
    except BaseException:
        logging.exception('Main eception')
        raise


if __name__ == '__main__':
    main()