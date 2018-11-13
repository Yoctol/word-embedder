from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np

from ..keyed_vectors import KeyedVectors
from .keyed_vectors_test_template import KeyedVectorsTestTemplate


ROOT_DIR = dirname(abspath(__file__))


class KeyedVectorsTestCase(KeyedVectorsTestTemplate, TestCase):

    def setUp(self):
        self.embedder = KeyedVectors(
            path=join(ROOT_DIR, 'data/example.vec'))
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

    def test_is_binary(self):
        self.assertFalse(self.embedder._binary)

    def test_path(self):
        self.assertEqual(
            join(ROOT_DIR, 'data/example.vec'),
            self.embedder._path,
        )


class KeyedVectorsBinTestCase(KeyedVectorsTestTemplate, TestCase):

    def setUp(self):
        self.embedder = KeyedVectors(
            path=join(ROOT_DIR, 'data/example.bin'),
            binary=True,
        )
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

    def test_is_binary(self):
        self.assertTrue(self.embedder._binary)

    def test_path(self):
        self.assertEqual(
            join(ROOT_DIR, 'data/example.bin'),
            self.embedder._path,
        )
