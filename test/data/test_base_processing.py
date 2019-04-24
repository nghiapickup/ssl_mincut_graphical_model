import unittest
import numpy as np

from data.newsgroups_processing import WordFeaturesScore, DocumentNormalizer


class TestNormalizer(unittest.TestCase):
    word_count_data = np.array([
            [4, 2,  10, 4,  2,  1,  1,  1,  3,  9],
            [2, 4,  0,  0,  0,  0,  1,  0,  0,  0],
            [0, 3,  3,  0,  1,  1,  1,  0,  8,  5],
            [0, 2,  0,  0,  0,  0,  0,  0,  4,  0],
            [0, 8,  8,  0,  0,  0,  1,  0,  6,  1],
            [0, 2,  1,  0,  0,  0,  0,  0,  4,  5]
        ])

    labels = np.array([2, 2, 0, 0, 1, 1])

    mi_rank_ids = np.array([0, 8, 2, 9, 1, 4, 7, 3, 5, 6])

    def test_WordFeaturesScore(self):
        # occurrences
        # word 0: [0, 2, 4]
        # word 1: [2, 3, 4, 8]
        # word 2: [0, 1, 3, 8, 10]
        # word 3: [0, 4]
        # word 4: [0, 1, 2]
        # word 5: [0, 1]
        # word 6: [0, 1]
        # word 7: [0, 1]
        # word 8: [0, 3, 4, 6, 8]
        # word 9: [0, 1, 5, 9]

        # P_x_y count
        # class\occurrences
        #       0, 2, 4| 2, 3, 4, 8| 0, 1, 3, 8, 10| 0, 4| 0, 1, 2| 0, 1| 0, 1| 0, 1| 0, 3, 4, 6, 8| 0, 1, 5, 9| P(y)
        #       -----------------------------------------------------------------------------------------------|-----
        # 0     2, 0, 0| 1, 1, 0, 0| 1, 0, 1, 0,  0| 2, 0| 1, 1, 0| 1, 1| 1, 1| 2, 0| 0, 0, 1, 0, 1| 1, 0, 1, 0| 20
        # 1     2, 0, 0| 1, 0, 0, 1| 0, 1, 0, 1,  0| 2, 0| 2, 0, 0| 2, 0| 1, 1| 2, 0| 0, 0, 1, 1, 0| 0, 1, 1, 0| 20
        # 2     0, 1, 1| 1, 0, 1, 0| 1, 0, 0, 0,  1| 1, 1| 1, 0, 1| 1, 1| 0, 2| 1, 1| 1, 1, 0, 0, 0| 1, 0, 0, 1| 20
        # P(x) [4, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1,  1, 5, 1, 4, 1, 1, 4, 2, 2, 4, 5, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1]

        p_x_y = np.array([[2, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0,  0, 2, 0, 1, 1,
                           0, 1, 1, 1, 1, 2, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0],
                          [2, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,  0, 2, 0, 2, 0,
                           0, 2, 0, 1, 1, 2, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
                          [0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0,  1, 1, 1, 1, 0,
                           1, 1, 1, 0, 2, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1]])
        p_x = np.array([4, 1, 1, 3, 1, 1, 1, 2, 1, 1, 1,  1, 5, 1, 4, 1,
                        1, 4, 2, 2, 4, 5, 1, 1, 1, 2, 1, 1, 2, 1, 2, 1])
        p_y = np.array([20, 20, 20])

        p_x_y = p_x_y/60
        p_x = p_x/60
        p_y = p_y/60
        p_x_p_y = np.dot(p_y.reshape((3, 1)), p_x.reshape(1, 32))

        mi_x_y = p_x_y * np.log((p_x_y + 1) / (p_x_p_y +1))

        mi_word = np.array([mi_x_y.T[0:3].sum(), mi_x_y.T[3:7].sum(),
                             mi_x_y.T[7:12].sum(), mi_x_y.T[12:14].sum(),
                             mi_x_y.T[14:17].sum(), mi_x_y.T[17:19].sum(),
                             mi_x_y.T[19:21].sum(), mi_x_y.T[21:23].sum(),
                             mi_x_y.T[23:28].sum(), mi_x_y.T[28:].sum(),])
        # [0 8 2 9 1 4 7 3 5 6]
        expected_mi_rank = np.argsort(mi_word)[::-1][:10]

        scorer = WordFeaturesScore(x_word_count=self.word_count_data, y=self.labels)
        scorer.scorer('mi')
        self.assertTrue(np.array_equal(expected_mi_rank, scorer._feature_rank_ids))

    def test_DocumentNormalizer(self):
        expected_x = np.array([[4, 3, 10, 9, 2],
                               [2, 0, 0,  0, 4],
                               [0, 8, 3,  5, 3],
                               [0, 4, 0,  0, 2],
                               [0, 6, 8,  1, 8],
                               [0, 4, 1,  5, 2]])
        l1_sum = np.array([4+3+10+9+2, 2+0+0+0+4, 0+8+3+5+3, 0+4+0+0+2, 0+6+8+1+8, 0+4+1+5+2])
        expected_x = (expected_x/l1_sum[:, np.newaxis]) * 10


        normalizer = DocumentNormalizer(feature_ids=self.mi_rank_ids[:5])
        x = normalizer.normalize('l1', self.word_count_data, scale=10)

        self.assertTrue(np.array_equal(expected_x, x))
