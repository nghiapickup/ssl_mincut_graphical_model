"""
    @nghia nh
    ---
    Preprocessing 20newsgroups data
"""

import logging

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from data.base_processing import BaseDataProcessor


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
        # data from basic slit train set, not actual working train set
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


class NewsgroupsData(BaseDataProcessor):
    """
    The origin data is loaded from sklearn auto fetch func

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
    """
    default_file_folder = 'origin/'
    test_file_folder = 'test/'

    def __init__(self, folder_dir, categories=None, positive_labels=None,
                 feature_score=None, feature_number=600, normalize=None, **kwargs_normalize):
        logging.info('NewsgroupsData __init__')

        self._newsgroups = fetch_20newsgroups(data_home=folder_dir,
                                             subset='all',
                                             categories=categories,
                                             shuffle=True,
                                             random_state=0,
                                             remove=('headers', 'footers', 'quotes'),
                                             download_if_missing=False)
        # using for feature selection
        self._newsgroups_train = fetch_20newsgroups(data_home=folder_dir,
                                                    subset='train',
                                                    categories=categories,
                                                    shuffle=True,
                                                    random_state=0,
                                                    remove=('headers', 'footers', 'quotes'),
                                                    download_if_missing=False)
        # load dict from a fixed list, this makes sure the same indices
        # between subset='all' and 'train' (which is used for feature reduction)
        self._dictionary = np.loadtxt(folder_dir + 'vocabulary.txt', dtype=str)
        self.positive_labels = positive_labels
        self.feature_score = feature_score
        self.feature_number = feature_number
        self.normalize = normalize
        self.kwargs_normalize = kwargs_normalize

        BaseDataProcessor.__init__(self)
        self._x_number = len(self._newsgroups.data)

        # the indices are fixed in data.target_names
        self._class_names = self._newsgroups.target_names

    def transform(self):
        logging.info('NewsgroupsData transform')

        self._y = self._newsgroups.target

        word_count_vectorizer = CountVectorizer(vocabulary=self._dictionary, stop_words='english')
        word_count_vectorizer._validate_vocabulary()

        # feature selection
        # To save calc time, this only works on default train data
        # (not train data at each cv fold)
        x_word_count_train = word_count_vectorizer.transform(self._newsgroups_train.data).toarray()
        feature_scorer = WordFeaturesScore(x_word_count_train, self._y)
        feature_scorer.scorer(self.feature_score)

        # normalize data
        x_word_count = word_count_vectorizer.transform(self._newsgroups.data).toarray()
        normalizer = DocumentNormalizer(feature_scorer.get_rank_ids(self.feature_number))
        self._x = normalizer.normalize(self.normalize, x_word_count, **self.kwargs_normalize)

        # binary class binding
        positive_label_indices = {self._class_names.index(label)
                                  for label in self.positive_labels}
        self._y = np.fromiter((x in positive_label_indices for x in self._y), dtype=int)

        return self
