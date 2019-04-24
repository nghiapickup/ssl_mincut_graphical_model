"""
    @nghia nh
    ---
    Preprocessing Reuters 21578 data
"""

import logging

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from data.reuters.fetch_reuters import get_reuters, fetch_data
from data.base_processing import BaseDataProcessor
from data.base_processing import WordFeaturesScore, DocumentNormalizer


class ReutersData(BaseDataProcessor):
    default_folder = 'reuters/data/'

    def __init__(self, folder_dir, positive_labels='acq',
                 feature_score='tfidf', feature_number=600,
                 normalize='tfidf', **kwargs_normalize):
        """
        Number of instances: 19716

        :param folder_dir: data folder
        :param positive_labels: categorizes that will be used for positive label
        :param feature_score: word scoring method for reducing feature
        :param feature_number: number of feature used (select from top words with high scrore)
        :param normalize: data normalize method (after feature reducing)
        """
        logging.info('ReutersData __init__')

        BaseDataProcessor.__init__(self)

        self._x_text, self._y = get_reuters(
            folder_dir=folder_dir,
            positive_labels=positive_labels)

        # load dict from a fixed list, this makes sure the same indices
        # between subset='all' and 'train' (which is used for feature reduction)
        self.positive_labels = positive_labels
        self.feature_score = feature_score
        self.feature_number = feature_number
        self.normalize = normalize
        self.kwargs_normalize = kwargs_normalize
        self._x_number = len(self._x_text)

    def _fetch_data(self, data_folder):
        """
        fetch Reuters-21578 data.
        :param data_folder: the dir to folder that store all data
        :return:
        """
        fetch_data(data_folder + self.default_folder)

    def transform(self):
        logging.info('ReutersData transform')

        word_count_vectorizer = CountVectorizer(stop_words='english')

        # feature selection
        # To saving calc time, this only works on a default data
        # (not the train data at each cv fold)
        x_word_count = word_count_vectorizer.fit_transform(self._x_text).toarray()
        feature_scorer = WordFeaturesScore(x_word_count, self._y)
        feature_scorer.scorer(self.feature_score)

        # normalize data
        normalizer = DocumentNormalizer(feature_scorer.get_rank_ids(self.feature_number))
        self._x = normalizer.normalize(self.normalize, x_word_count, **self.kwargs_normalize)
        self._y = np.array(self._y)

        return self
