import io
from os.path import isfile, basename
import os

import numpy as np

from .fasttext import FastText
from .oov_error import OOVError
from .utils import download_data, extract_gz


class FastTextLight(FastText):

    def __init__(self, path: str):
        super().__init__(path=path)

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
            ) = self._load_data(path=self._path)

            self._vloader = LowMemoryVecLoader(path=self._path)
            self._is_built = True

    def _get_vector(self, index: int) -> np.ndarray:
        if (index >= 0) and (index < self._vocab_size):
            return self._vloader[index]
        else:
            raise OOVError

    @staticmethod
    def _load_data(path: str):
        fin = io.open(
            path, 'r',
            encoding='utf-8',
            newline='\n',
            errors='ignore',
        )
        vocab_size, embedding_size = map(int, fin.readline().split())

        vocab_list = ['0'] * vocab_size

        for idx, line in enumerate(fin):
            tokens = line.rstrip().split(' ')
            vocab_list[idx] = tokens[0]
        fin.close()
        return embedding_size, vocab_size, vocab_list

    def __del__(self):
        if '_vloader' in self.__dict__:
            del self._vloader


class LowMemoryVecLoader:

    def __init__(self, path: str):
        self.fin = io.open(
            path, 'r',
            encoding='utf-8',
            newline='\n',
            errors='ignore',
        )
        self._init_fin()

    def _init_fin(self):
        self.fin.seek(0)  # reset fin
        # skip first line
        self._vocab_size, _ = map(int, self.fin.readline().split())

        self.current_pt = 0
        self.current_vector = self._get_vector()

    def _skip_lines(self, num: int = 1):
        for _ in range(num):
            self.fin.readline()

    def _get_vector(self) -> np.ndarray:
        line = self.fin.readline()
        tokens = line.rstrip().split(' ')
        vector = list(map(float, tokens[1:]))
        return np.array(vector).astype(np.float32)

    def __getitem__(self, index: int):
        if (index < 0) or (index > self._vocab_size):
            raise ValueError('Out of index')

        if index < self.current_pt:
            self._init_fin()

        if index == self.current_pt:
            return self.current_vector

        diff = index - self.current_pt - 1
        self._skip_lines(diff)
        self.current_pt += diff + 1
        self.current_vector = self._get_vector()
        return self.current_vector

    def __del__(self):
        self.fin.close()
        del self.fin
