from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np

from ..fasttext import FastText
from ..oov_error import OOVError


ROOT_DIR = dirname(abspath(__file__))


class FastTextTestTemplate:

    def test_correctly_create_instance(self):
        self.assertEqual(
            set(['_path', '_is_built']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            join(ROOT_DIR, 'data/fasttext.vec'),
            self.embedder._path,
        )
        self.assertFalse(self.embedder._is_built)

    def test_build(self):
        self.embedder.build()
        self.assertTrue(self.embedder._is_built)
        self.assertEqual(
            set(['_path', '_is_built',
                 '_embedding_size', '_vocab_size',
                 '_word_vectors', '_vocab_list']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(self.words, self.embedder._vocab_list)
        self.assertEqual(
            self.vectors.tolist(),
            self.embedder._word_vectors.tolist(),
        )

    def test_vocab_size(self):
        self.embedder.build()
        self.assertEqual(len(self.words), self.embedder.n_vocab)

    def test_n_dim(self):
        self.embedder.build()
        self.assertEqual(self.vectors.shape[1], self.embedder.n_dim)
        self.assertEqual(
            self.embedder._embedding_size,
            self.embedder.n_dim,
        )

    def test_get_index(self):
        self.embedder.build()
        self.assertEqual(2, self.embedder.get_index('gb'))

    def test_get_index_oov(self):
        self.embedder.build()
        self.assertEqual(-1, self.embedder.get_index('haha'))

    def test_get_word(self):
        self.embedder.build()
        self.assertEqual('薄餡', self.embedder.get_word(0))

    def test_get_word_oov(self):
        self.embedder.build()
        self.assertIsNone(self.embedder.get_word(10))

    def test_getitem_string(self):
        self.embedder.build()
        for i in range(len(self.words)):
            with self.subTest(i=i):
                self.assertEqual(
                    self.vectors[i].tolist(),
                    self.embedder[self.words[i]].tolist(),
                )

    def test_getitem_int(self):
        self.embedder.build()
        for i in range(len(self.words)):
            with self.subTest(i=i):
                self.assertEqual(
                    self.vectors[i].tolist(),
                    self.embedder[i].tolist(),
                )

    def test_getitem_string_oov(self):
        self.embedder.build()
        with self.assertRaises(OOVError):
            self.embedder['kerker']

    def test_getitem_int_oov(self):
        self.embedder.build()
        with self.assertRaises(OOVError):
            self.embedder[100]

    def test_getitem_wrong_type(self):
        self.embedder.build()
        with self.assertRaises(TypeError):
            self.embedder[12.3]
            self.embedder[[123]]


class FastTextTestCase(FastTextTestTemplate, TestCase):

    def setUp(self):
        self.embedder = FastText(
            path=join(ROOT_DIR, 'data/fasttext.vec'))
        self.words = ['薄餡', '隼興', 'gb', 'en', 'Alvin']
        self.vectors = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.11, 0.12, 0.13],
                [0.14, 0.15, 0.16],
            ],
        ).astype(np.float32)
