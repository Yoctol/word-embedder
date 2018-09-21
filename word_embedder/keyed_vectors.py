import io
from os.path import isfile, basename
import os

import numpy as np

from .base import BaseEmbedder
from .oov_error import OOVError
from .utils import download_data, extract_gz


def _load_text_file(path: str):
    fin = io.open(
        path, 'r',
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


def _load_bin_file(path: str):
    # load .bin file
    # Note that float in this file should be float32
    # float64 is not allowed
    fin = open(path, 'rb')
    header = fin.readline().decode('utf8')
    vocab_size, embedding_size = (int(x) for x in header.split())

    # init vocab list
    vocab_list = ['0'] * vocab_size
    word_vectors = np.random.rand(
        vocab_size, embedding_size).astype(np.float32)

    binary_len = 4 * embedding_size  # float32
    for idx in range(vocab_size):
        # mixed text and binary: read text first, then binary
        word = []
        while True:
            ch = fin.read(1)
            if ch == b' ':
                break
            if ch == b'':
                raise EOFError(
                    "unexpected end of input; is count incorrect or file otherwise damaged?")
            # ignore newlines in front of words (some binary files have)
            if ch != b'\n':
                word.append(ch)

        vocab_list[idx] = b''.join(word).decode('utf8')
        word_vectors[idx, :] = np.frombuffer(
            fin.read(binary_len),
            dtype='float32',
        )
    fin.close()
    return embedding_size, vocab_size, vocab_list, word_vectors


class KeyedVectors(BaseEmbedder):

    def __init__(self, path: str, binary: bool=False):
        self._path = path
        self._binary = binary
        self._is_built = False

    def build(self):
        if not self._is_built:
            if not isfile(self._path):
                # if data is not at self._path
                # download it through url in .env
                download_data(
                    url=os.getenv(basename(self._path)),
                    output_path=self._path + '.gz',
                )
                extract_gz(self._path + '.gz')
            (
                self._embedding_size,
                self._vocab_size,
                self._vocab_list,
                self._word_vectors,
            ) = self._load_data(
                path=self._path,
                binary=self._binary,
            )
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
    def _load_data(path: str, binary: bool=False):
        if binary:
            return _load_bin_file(path=path)
        else:
            return _load_text_file(path=path)
