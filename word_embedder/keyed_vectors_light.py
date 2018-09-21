from typing import List
import io
from os.path import isfile, basename
import os

import numpy as np

from .keyed_vectors import KeyedVectors
from .oov_error import OOVError
from .utils import download_data, extract_gz


def _load_text_file(path: str):
    """Load .vec file"""
    fin = io.open(path, 'rb')
    first_line = fin.readline().decode('utf8')
    vocab_size, embedding_size = map(int, first_line.split())

    # init vocab list
    vocab_list = ['0'] * vocab_size

    # record start position of each line in file
    byte_pos = [0] * (vocab_size + 1)
    byte_pos[0] = fin.tell()

    for idx in range(vocab_size):
        line = fin.readline()
        tokens = line.rstrip().split(b' ')
        vocab_list[idx] = tokens[0].decode('utf8')
        byte_pos[idx + 1] = fin.tell()
    fin.close()
    return embedding_size, vocab_size, vocab_list, byte_pos


def _load_bin_file(path: str):
    # load .bin file
    # Note that float in this file should be float32
    # float64 is not allowed
    fin = open(path, 'rb')
    header = fin.readline().decode('utf8')
    vocab_size, embedding_size = (int(x) for x in header.split())

    # init vocab list
    vocab_list = ['0'] * vocab_size

    # record start position of each line in file
    byte_pos = [0] * vocab_size

    binary_len = 4 * embedding_size  # 4: because of float32
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

        byte_pos[idx] = fin.tell()
        vocab_list[idx] = b''.join(word).decode('utf8')
        fin.read(binary_len)
    fin.close()
    return embedding_size, vocab_size, vocab_list, byte_pos


class KeyedVectorsLight(KeyedVectors):

    def __init__(self, path: str, binary: bool=False):
        super().__init__(path=path, binary=binary)

    def build(self):
        if self._is_built:
            pass

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
        ) = self._load_data(
            path=self._path,
            binary=self._binary,
        )

        self._vloader = LowMemoryVecLoader(
            path=self._path,
            byte_pos=self._byte_pos,
            binary=self._binary,
        )
        self._is_built = True

    def _get_vector(self, index: int) -> np.ndarray:
        if (index >= 0) and (index < self._vocab_size):
            return self._vloader[index]
        else:
            raise OOVError

    @staticmethod
    def _load_data(path: str, binary: bool=False):
        if binary:
            return _load_bin_file(path=path)
        else:
            return _load_text_file(path=path)

    def __del__(self):
        if '_vloader' in self.__dict__:
            del self._vloader


class LowMemoryVecLoader:

    def __init__(
            self,
            path: str,
            byte_pos: List[int],
            binary: bool = False,
        ):
        self._byte_pos = byte_pos
        self._binary = binary

        self.fin = open(path, 'rb')
        header = self.fin.readline().decode('utf8')
        self._vocab_size, self._embedding_size = map(int, header.split())

    def _get_vector_from_text(self, index: int) -> np.ndarray:
        # get start end position from _byte_pos
        start_pos = self._byte_pos[index]
        end_pos = self._byte_pos[index + 1]

        #
        diff = end_pos - start_pos

        self.fin.seek(start_pos, 0)
        line = self.fin.read(diff).decode('utf8')

        tokens = line.rstrip().split(' ')
        vector = list(map(float, tokens[1:]))
        return np.array(vector).astype(np.float32)

    def _get_vector_from_bin(self, index: int) -> np.ndarray:
        binary_len = 4 * self._embedding_size  # 4: because of float32

        start_pos = self._byte_pos[index]

        self.fin.seek(start_pos, 0)

        vector = np.frombuffer(
            self.fin.read(binary_len),
            dtype='float32',
        )
        return vector

    def __getitem__(self, index: int) -> np.ndarray:
        if (index < 0) or (index > self._vocab_size):
            raise ValueError('Out of index')

        if self._binary:
            vector = self._get_vector_from_bin(index=index)
        else:
            vector = self._get_vector_from_text(index=index)
        return vector

    def __del__(self):
        self.fin.close()
        del self.fin
