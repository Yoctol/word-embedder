from typing import List
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
                self._byte_pos,
            ) = self._load_data(path=self._path)

            self._vloader = LowMemoryVecLoader(
                path=self._path,
                byte_pos=self._byte_pos,
            )
            self._is_built = True

    def _get_vector(self, index: int) -> np.ndarray:
        if (index >= 0) and (index < self._vocab_size):
            return self._vloader[index]
        else:
            raise OOVError

    @staticmethod
    def _load_data(path: str):
        fin = io.open(path, 'rb')
        first_line = fin.readline().decode('utf8')
        vocab_size, embedding_size = map(int, first_line.split())

        vocab_list = ['0'] * vocab_size
        byte_pos = [0] * (vocab_size + 1)
        byte_pos[0] = fin.tell()

        for idx in range(vocab_size):
            line = fin.readline()
            tokens = line.rstrip().split(b' ')
            vocab_list[idx] = tokens[0].decode('utf8')
            byte_pos[idx + 1] = fin.tell()
        fin.close()
        return embedding_size, vocab_size, vocab_list, byte_pos

    def __del__(self):
        if '_vloader' in self.__dict__:
            del self._vloader


class LowMemoryVecLoader:

    def __init__(self, path: str, byte_pos: List[int]):
        self.fin = io.open(path, 'rb')
        first_line = self.fin.readline().decode('utf8')
        self._vocab_size, _ = map(int, first_line.split())
        self._byte_pos = byte_pos

    def _get_line(self, index: int) -> str:
        start_pos = self._byte_pos[index]
        end_pos = self._byte_pos[index + 1]
        diff = end_pos - start_pos
        self.fin.seek(0)
        self.fin.seek(start_pos)
        line = self.fin.read(diff).decode('utf8')
        return line

    def _get_vector(self, line: str) -> np.ndarray:
        tokens = line.rstrip().split(' ')
        vector = list(map(float, tokens[1:]))
        return np.array(vector).astype(np.float32)

    def __getitem__(self, index: int):
        if (index < 0) or (index > self._vocab_size):
            raise ValueError('Out of index')

        line = self._get_line(index=index)
        vector = self._get_vector(line=line)
        return vector

    def __del__(self):
        self.fin.close()
        del self.fin
