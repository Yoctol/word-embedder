from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np

from ..keyed_vectors_light import KeyedVectorsLight
from .keyed_vectors_test_template import KeyedVectorsTestTemplate

ROOT_DIR = dirname(abspath(__file__))


class KeyedVectorsLightTestCase(KeyedVectorsTestTemplate, TestCase):

    def setUp(self):
        self.embedder = KeyedVectorsLight(
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

    def test_correctly_create_instance(self):
        self.assertEqual(
            set(['_path', '_binary', '_is_built']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            join(ROOT_DIR, 'data/example.vec'),
            self.embedder._path,
        )
        self.assertFalse(self.embedder._binary)
        self.assertFalse(self.embedder._is_built)

    def test_build(self):
        self.embedder.build()
        self.assertTrue(self.embedder._is_built)
        self.assertEqual(
            set(['_path', '_binary', '_is_built',
                 '_embedding_size', '_vocab_size',
                 '_vocab_list', '_byte_pos',
                 '_vloader']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            ['薄餡', '隼興', 'gb', 'en', 'Alvin'],
            self.embedder._vocab_list,
        )


class KeyedVectorsLightBinTestCase(KeyedVectorsTestTemplate, TestCase):

    def setUp(self):
        self.embedder = KeyedVectorsLight(
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

    def test_correctly_create_instance(self):
        self.assertEqual(
            set(['_path', '_binary', '_is_built']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            join(ROOT_DIR, 'data/example.bin'),
            self.embedder._path,
        )
        self.assertTrue(self.embedder._binary)
        self.assertFalse(self.embedder._is_built)

    def test_build(self):
        self.embedder.build()
        self.assertTrue(self.embedder._is_built)
        self.assertEqual(
            set(['_path', '_binary', '_is_built',
                 '_embedding_size', '_vocab_size',
                 '_vocab_list', '_byte_pos',
                 '_vloader']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            ['薄餡', '隼興', 'gb', 'en', 'Alvin'],
            self.embedder._vocab_list,
        )
