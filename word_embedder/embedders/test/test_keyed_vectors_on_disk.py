from unittest import TestCase
from os.path import abspath, dirname, join, isfile

import numpy as np
import pickle as pkl

from ..keyed_vectors_on_disk import KeyedVectorsOnDisk
from .keyed_vectors_test_template import KeyedVectorsTestTemplate

ROOT_DIR = dirname(abspath(__file__))


class KeyedVectorsOnDiskTestCase(KeyedVectorsTestTemplate, TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = join(ROOT_DIR, 'data/example_on_disk.bin')
        cls.array_path = join(ROOT_DIR, 'data/example_on_disk.ny')
        cls.vectors = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.11, 0.12, 0.13],
                [0.14, 0.15, 0.16],
            ],
        ).astype(np.float32)
        cls.words = ['薄餡', '隼興', 'gb', 'en', 'Alvin']
        cls.param = {
            'vocab': None,
            'index2word': cls.words,
            'vector_size': cls.vectors.shape[1],
            'syn0shape': cls.vectors.shape,
            'syn0dtype': cls.vectors.dtype,
            'syn0filename': cls.array_path,
        }

        if not isfile(cls.array_path):
            fp = np.memmap(
                cls.array_path,
                dtype=cls.vectors.dtype,
                mode='w+',
                shape=cls.vectors.shape,
            )
            fp[:] = cls.vectors[:]
            del fp

            with open(cls.path, 'wb') as f:
                pkl.dump(cls.param, f, protocol=pkl.HIGHEST_PROTOCOL)

    def setUp(self):
        self.embedder = KeyedVectorsOnDisk(
            path=self.path,
        )

    def test_correctly_create_instance(self):
        self.assertEqual(
            set(['_path', '_array_path', '_is_built']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            join(ROOT_DIR, 'data/example_on_disk.bin'),
            self.embedder._path,
        )
        self.assertFalse(self.embedder._is_built)

    def test_build(self):
        self.embedder.build()
        self.assertTrue(self.embedder._is_built)
        self.assertEqual(
            set(['_path', '_array_path', '_is_built',
                 '_embedding_size', '_vocab_size',
                 '_vocab_list', '_word_vectors']),
            set(self.embedder.__dict__.keys()),
        )
        # check words
        self.assertEqual(self.words, self.embedder._vocab_list)


class KeyedVectorsOnDiskGivenArrayPathTestCase(
        KeyedVectorsTestTemplate, TestCase):

    @classmethod
    def setUpClass(cls):
        cls.path = join(ROOT_DIR, 'data/example_on_disk.bin')
        cls.array_path_old = join(ROOT_DIR, 'data/example_on_disk.ny')
        cls.array_path = join(ROOT_DIR, 'data/example1_on_disk.ny')

        cls.vectors_old = np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9],
                [0.11, 0.12, 0.13],
                [0.14, 0.15, 0.16],
            ],
        ).astype(np.float32)
        cls.vectors = cls.vectors_old + 10
        cls.words = ['薄餡', '隼興', 'gb', 'en', 'Alvin']

        cls.param = {
            'vocab': None,
            'index2word': cls.words,
            'vector_size': cls.vectors.shape[1],
            'syn0shape': cls.vectors.shape,
            'syn0dtype': cls.vectors.dtype,
            'syn0filename': cls.array_path_old,
        }

        if not isfile(cls.array_path):
            for path, v in zip(
                [cls.array_path_old, cls.array_path],
                [cls.vectors_old, cls.vectors],
            ):
                fp = np.memmap(
                    path,
                    dtype=cls.vectors.dtype,
                    mode='w+',
                    shape=cls.vectors.shape,
                )
                fp[:] = v[:]
                del fp

            with open(cls.path, 'wb') as f:
                pkl.dump(cls.param, f, protocol=pkl.HIGHEST_PROTOCOL)

    def setUp(self):
        self.embedder = KeyedVectorsOnDisk(
            path=self.path,
            array_path=self.array_path,
        )

    def test_correctly_create_instance(self):
        self.assertEqual(
            set(['_path', '_array_path', '_is_built']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            join(ROOT_DIR, 'data/example_on_disk.bin'),
            self.embedder._path,
        )
        self.assertFalse(self.embedder._is_built)

    def test_build(self):
        self.embedder.build()
        self.assertTrue(self.embedder._is_built)
        self.assertEqual(
            set(['_path', '_array_path', '_is_built',
                 '_embedding_size', '_vocab_size',
                 '_vocab_list', '_word_vectors']),
            set(self.embedder.__dict__.keys()),
        )
        # check array path should be the one given
        # when embedder is initialized
        # not the one stored in param
        self.assertNotEqual(
            self.param['syn0filename'], self.embedder._array_path)
        self.assertEqual(
            self.array_path, self.embedder._array_path)

        # check words
        self.assertEqual(self.words, self.embedder._vocab_list)

    def test_vectors_not_from_param(self):
        self.embedder.build()
        for i, word in enumerate(self.words):
            self.assertNotEqual(
                self.param['syn0filename'][i],  # vectors stored in path
                self.embedder[word].tolist(),  # vectors stored in array_path
            )
