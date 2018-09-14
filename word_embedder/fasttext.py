import io

import numpy as np

from .base_embedder import BaseEmbedder
from .oov_error import OOVError


class FastText(BaseEmbedder):

    def __init__(self, path: str):
        self._path = path
        self._is_built = False

    def build(self):
        if not self._is_built:
            (
                self._embedding_size,
                self._vocab_size,
                self._vocab_list,
                self._word_vectors,
            ) = self._load_data(fname=self._path)
            self._is_built = True

    def __getitem__(self, key) -> np.ndarray:
        """Get a word vector

            If key is an int, return vector by index.
            If key is a string, return vector by word.

        """
        if isinstance(key, str):
            index = self.get_index(word=key)
        elif isinstance(key, int):
            index = key
        else:
            raise TypeError(
                'Only support int and str type of input',
            )
        vector = self._get_vector(index)
        return vector

    @property
    def n_vocab(self) -> int:
        """Vocabulary size"""
        return self._vocab_size

    @property
    def n_dim(self) -> int:
        """Embedding size"""
        return self._embedding_size

    def get_index(self, word: str) -> int:
        try:
            index = self._vocab_list.index(word)
        except ValueError:
            index = -1
        return index

    def get_word(self, index: int) -> str:
        word = None
        try:
            word = self._vocab_list[index]
        except IndexError:
            print(
                'index [{}] out of range (max vocab size = {})'.format(
                    index,
                    self._vocab_size,
                ),
            )
        return word

    def _get_vector(self, index: int) -> np.ndarray:
        if (index >= 0) and (index < self._vocab_size):
            return self._word_vectors[index, :]
        else:
            raise OOVError

    @staticmethod
    def _load_data(fname: str):
        fin = io.open(
            fname, 'r',
            encoding='utf-8',
            newline='\n',
            errors='ignore',
        )
        vocab_size, embedding_size = map(int, fin.readline().split())

        vocab_list = ['0'] * vocab_size
        word_vectors = np.random.rand(
            vocab_size, embedding_size).astype(np.float32)

        for idx, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            vocab_list[idx] = tokens[0]
            vector = list(map(float, tokens[1:]))
            word_vectors[idx, :] = np.array(vector).astype(np.float32)
        fin.close()
        return embedding_size, vocab_size, vocab_list, word_vectors
