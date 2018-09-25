import numpy as np
import pickle as pkl

from .keyed_vectors import KeyedVectors


class KeyedVectorsOnDisk(KeyedVectors):

    """

    Reference from "https://github.com/Yoctol/keyedvectorsondisk"
    input data should be a pickle file and
    with the format
    {
        'vocab': ['a', 'apple', ...],  # list of vocabs
        'index2word': self.index2word,  # optional
        'vector_size': 100,  # size of a word vector
        'syn0shape': self.syn0.shape,   # optional
        'syn0dtype': self.syn0.dtype,   # data type of word vectors
        'syn0filename': os.path.abspath(self.syn0.filename),  # path of
        # where to store word vectors,
    }

    """
    def __init__(
            self,
            path: str,
            array_path: str=None,
        ):
        self._path = path
        self._array_path = array_path
        self._is_built = False

    def build(self):
        if not self._is_built:
            (
                self._embedding_size,
                self._vocab_size,
                self._vocab_list,
                self._word_vectors,
            ) = self._load_data(
                path=self._path,
                array_path=self._array_path,
            )
            self._is_built = True

    @staticmethod
    def _load_data(path: str, array_path: str=None):
        with open(path, 'rb') as f:
            f.seek(0)
            param = pkl.load(f)

        vocab_list = param['index2word']
        embedding_size = param['vector_size']
        vocab_size = len(vocab_list)

        vector_path = param['syn0filename'] if array_path is None else array_path
        word_vectors = np.memmap(
            filename=vector_path,
            shape=(vocab_size, embedding_size),
            dtype=param['syn0dtype'],
            mode='r',
        )
        return embedding_size, vocab_size, vocab_list, word_vectors
