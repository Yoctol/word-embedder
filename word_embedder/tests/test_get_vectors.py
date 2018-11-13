from unittest import TestCase

import numpy as np

from ..get_vectors import (
    get_vectors,
    remove_duplicated_words,
)


class MockEmbedder(object):

    def __init__(self):
        self._vocab_list = ['記良', '隼興', '建甫', '勤彥', '祥瑞', '宜恩']
        self.vectors = np.array(
            [
                [1, 1, 1],
                [2, 2, 2],
                [3, 3, 3],
                [4, 4, 4],
                [5, 5, 5],
                [6, 6, 6],
            ],
        ).astype('float32')

    @property
    def n_dim(self):
        return self.vectors.shape[1]

    @property
    def vocab(self):
        return self._vocab_list

    def get_index(self, word: str):
        if word not in self._vocab_list:
            return -1
        else:
            return self._vocab_list.index(word)

    def __getitem__(self, key):
        return self.vectors[key, :]


class GetVectorsTestCase(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mocked_embedder = MockEmbedder()

    def test_remove_duplicated_words_normal_case(self):
        input_words = ['蘋果', '香蕉', '梨子']
        expected_output = ['蘋果', '香蕉', '梨子']
        output = remove_duplicated_words(input_words)
        self.assertEqual(expected_output, output)

    def test_remove_duplicated_words_duplicated(self):
        input_words = ['蘋果', '香蕉', '蘋果', '梨子']
        expected_output = ['蘋果', '香蕉', '梨子']
        output = remove_duplicated_words(input_words)
        self.assertEqual(expected_output, output)

    def test_get_vectors_normal_case(self):
        input_words = ['記良', '隼興', '建甫', '勤彥', '祥瑞', '宜恩']
        output = get_vectors(
            embedder=self.mocked_embedder,
            words=input_words,
        )
        expected_output = self.mocked_embedder.vectors
        np.testing.assert_array_equal(expected_output, output)

    def test_get_vectors_oov(self):
        input_words = ['建甫', '家豪']
        output = get_vectors(
            embedder=self.mocked_embedder,
            words=input_words,
        )
        np.testing.assert_array_equal(output[0], self.mocked_embedder.vectors[2, :])
        self.assertEqual((2, self.mocked_embedder.n_dim), output.shape)

    def test_get_vectors_duplicated(self):
        input_words = ['記良', '記良', '隼興', '記良', '建甫', '記良', '隼興']
        output = get_vectors(
            embedder=self.mocked_embedder,
            words=input_words,
        )
        expected_output = self.mocked_embedder.vectors[0: 3, ]
        np.testing.assert_array_equal(output, expected_output)
