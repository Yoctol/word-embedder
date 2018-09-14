from unittest import TestCase
from os.path import abspath, dirname, join

from ..fasttext_light import FastTextLight
from .test_fasttext import FastTextTestTemplate

ROOT_DIR = dirname(abspath(__file__))


class FastTextLightTestCase(FastTextTestTemplate, TestCase):

    def setUp(self):
        self.embedder = FastTextLight(
            path=join(ROOT_DIR, 'data/fasttext.vec'))

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
                 '_vocab_list', '_vloader']),
            set(self.embedder.__dict__.keys()),
        )
        self.assertEqual(
            ['薄餡', '隼興', 'gb', 'en', 'Alvin'],
            self.embedder._vocab_list,
        )
