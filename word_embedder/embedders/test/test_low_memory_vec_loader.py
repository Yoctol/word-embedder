from unittest import TestCase
from os.path import abspath, dirname, join

import numpy as np

from ..keyed_vectors_light import LowMemoryVecLoader


ROOT_DIR = dirname(abspath(__file__))


class LowMemoryVecLoaderTestCase(TestCase):

    def setUp(self):
        self.loader = LowMemoryVecLoader(
            path=join(ROOT_DIR, 'data/example.vec'),
            byte_pos=[4, 23, 42, 57, 75, 95],
        )

    def test_getitem_forward(self):
        self.assertEqual(
            np.array([0.4, 0.5, 0.6]).astype(np.float32).tolist(),
            self.loader[1].tolist(),
        )
        self.assertEqual(
            np.array([0.11, 0.12, 0.13]).astype(np.float32).tolist(),
            self.loader[3].tolist(),
        )

    def test_getitem_backward(self):
        self.assertEqual(
            np.array([0.14, 0.15, 0.16]).astype(np.float32).tolist(),
            self.loader[4].tolist(),
        )
        self.assertEqual(
            np.array([0.1, 0.2, 0.3]).astype(np.float32).tolist(),
            self.loader[0].tolist(),
        )

    def test_getitem_out_of_index(self):
        with self.assertRaises(ValueError):
            self.loader[100]
