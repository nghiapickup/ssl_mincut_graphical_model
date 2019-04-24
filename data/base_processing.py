"""
    @nghia nh
    ---
    Class base for data processing and support functions


"""
import  logging

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


class BaseDataProcessor:
    """
    This is a common interface processing raw data.
    Giving match output for input of Pipeline's data constructor or classifier

    This form implies that we handle all the data and
    only support getting train - test split from provided indices.

    The process follows:
    reading draw data -> <transform> into required type ->
    get splitting indices(using outside data tool) -> <extract> data from indices.
    """
    def __init__(self):
        """
        Reading raw data and init some basic information
        """
        self._x = None
        self._y = None

        # data number prepares for splitter using a placeholder for actual data
        # (e.g. train-test split indices form sklearn split)
        self._x_number = 0

    @property
    def x_number(self):
        """
        prevent messing this property.
        x_number and y are required for getting split indices.
        :return:
        """
        return self._x_number

    @property
    def y(self):
        """
        prevent messing this property.
        x_number and y are required for getting split indices.
        :return:
        """
        return self._y

    def transform(self):
        """
        This should set the values for self.x and self.y
        as all data features and corresponding labels.
        Also we transform data into required form (e.g. binary, multi-class)
        and overall processing which does not affect the un-touch rule for test data (in future).
        :return:
        """
        return self

    def extract(self, train_indices, test_indices):
        """
        Extract data when we have train and test split indices.
        We should process normalizing here and this mus be processing ONLY on train set
        then apply the same setting for test data.
        :param train_indices: train data indices
        :param test_indices: test data indices
        :return:
        """
        return self._x[train_indices], self._y[train_indices], self._x[test_indices], self._y[test_indices]


###############################################################################
# Text feature processing

class WordFeaturesScore:
    def __init__(self, x_word_count, y):
        self._x_word_count = x_word_count
        self._y = y
        self._feature_rank_ids = np.arange(self._x_word_count.shape[1])

        self._name_map = {
            'mi': self._mutual_information_score,
            'tfidf': self._tfidf_score
        }

    def get_rank_ids(self, n_features):
        return self._feature_rank_ids[:n_features]

    def scorer(self, score_name):
        try:
            return self._name_map[score_name]()
        except KeyError:
            logging.error('WordFeaturesScore: None supported scorer')
            raise

    def _mutual_information_score(self):
        """
        Feature selection (reduction) using Average Mutual Information of
        X: Word occurrences (a vocabulary has many occurrences)
        Y: Class of document of a word occurrences (a word's occurrence may appear in many classes)
        """

        # dict length gets directly from train data to feasible with unittest
        dict_length = len(self._x_word_count.T)
        class_number = len(self._y)

        # we only need to calc the joint pr P(Y, X) table,
        # the marginal P(X), P(Y) can be sum up by row, column.
        # Reminder that 0 (word does not appear every where) is still an occurrence

        p_x_y_by_word = []
        for word_id in range(dict_length):
            occurs, inverse_occur = np.unique(self._x_word_count.T[word_id], return_inverse=True)
            occurs_count = np.zeros((class_number, len(occurs)))
            for doc_id, occur in enumerate(inverse_occur):
                occurs_count[self._y[doc_id], occur] += 1
            p_x_y_by_word.append(occurs_count)
        p_x_y = np.hstack(p_x_y_by_word)
        p_y = p_x_y.sum(axis=1)
        sum_occurs = np.sum(p_y)
        p_y = p_y/sum_occurs

        # We calc P_x_y by word, it make easier to sum up MI of a word
        mi_list = []
        for p_x_y in p_x_y_by_word:
            p_x_y = p_x_y / sum_occurs
            p_x_by_word = p_x_y.sum(axis=0)
            p_x_p_y_by_word = np.dot(p_y.reshape(class_number,1),
                                     p_x_by_word.reshape(1,len(p_x_by_word))) # P(x)P(y)
            # P(x,y) * log( P(x,y) / (P(x)*P(y)) )
            # If P(x) or P(y) =0 then P(x,y)=0
            mi = p_x_y * np.log((p_x_y+1)/(p_x_p_y_by_word+1))
            mi_list.append(np.sum(mi))

        self._feature_rank_ids = np.argsort(np.array(mi_list))[::-1][:dict_length]

    def _tfidf_score(self):
        # data from basic split of the train set, not the actual working train set
        vectorize = TfidfTransformer()
        data = vectorize.fit_transform(self._x_word_count).toarray()
        data = np.sum(data, axis=0)

        self._feature_rank_ids = np.argsort(data)[::-1][:len(data.T)]


class DocumentNormalizer:
    def __init__(self, feature_ids):
        self._feature_ids = feature_ids

        self._name_map = {
            'l1': self._l1_normalizer,
            'tfidf': self._tfidf_normalizer
        }

    def normalize(self, normalize_name, x_word_count, **kwargs):
        try:
            return self._name_map[normalize_name](x_word_count, **kwargs)
        except KeyError:
            logging.error('DocumentNormalizer: None supported normalizer')
            raise

    def _l1_normalizer(self, x_word_count, scale=10000.):
        selected_data = x_word_count[:, self._feature_ids]
        vectorizer = TfidfTransformer(norm='l1', use_idf=False)
        data =  vectorizer.transform(selected_data).toarray() # get tf
        return scale * data

    def _tfidf_normalizer(self, x_word_count, **kwargs):
        selected_data = x_word_count[:, self._feature_ids]
        vectorizer = TfidfTransformer(norm='l2', use_idf=True)
        return vectorizer.fit_transform(selected_data).toarray()
