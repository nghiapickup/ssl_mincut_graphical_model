"""
    @nghia nh
    ---
    Preprocessing 20newsgroups data
"""

import logging

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

from data.base_processing import BaseDataProcessor
from data.base_processing import WordFeaturesScore, DocumentNormalizer


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
    default_folder = '20newsbydate/'

    def __init__(self, folder_dir, categories=None, positive_labels=None,
                 feature_score='tfidf', feature_number=600,
                 normalize='tfidf', **kwargs_normalize):
        """
        :param folder_dir: data folder
        :param categories: picked categorizes
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        """
        logging.info('NewsgroupsData __init__')

        BaseDataProcessor.__init__(self)

        self._newsgroups = fetch_20newsgroups(
            data_home=folder_dir,
            subset='all',
            categories=categories,
            shuffle=True,
            random_state=0,
            remove=('headers', 'footers', 'quotes'),
            download_if_missing=False
        )
        # use for feature selection
        self._newsgroups_train = fetch_20newsgroups(
            data_home=folder_dir,
            subset='train',
            categories=categories,
            shuffle=True,
            random_state=0,
            remove=('headers', 'footers', 'quotes'),
            download_if_missing=False
        )

        # load dict from a fixed list, this makes sure the same indices
        # between subset='all' and 'train' (which is used for feature reduction)
        self._dictionary = np.loadtxt(folder_dir + 'vocabulary.txt', dtype=str)
        self.positive_labels = positive_labels
        self.feature_score = feature_score
        self.feature_number = feature_number
        self.normalize = normalize
        self.kwargs_normalize = kwargs_normalize
        self._x_number = len(self._newsgroups.data)

        # the indices are fixed in data.target_names
        self._class_names = self._newsgroups.target_names

    def transform(self):
        logging.info('NewsgroupsData transform')

        self._y = self._newsgroups.target

        word_count_vectorizer = CountVectorizer(vocabulary=self._dictionary, stop_words='english')
        word_count_vectorizer._validate_vocabulary()

        # feature selection
        # To saving calc time, this only works on a default train data
        # (not the train data at each cv fold)
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
