from abc import abstractmethod
from typing import Dict

import numpy as np


class BaseEmbedder:

    @abstractmethod
    def __getitem__(self, key) -> np.ndarray:
        """Get a word vector

            If key is an int, return vector by index.
            If key is a string, return vector by word.

        """
        raise NotImplementedError

    @abstractmethod
    def vocab_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def n_dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_index(self, word: str) -> int:
        """Return word index
        """
        raise NotImplementedError

    @abstractmethod
    def get_word(self, index: int) -> str:
        """Return word
        """
        raise NotImplementedError

    @abstractmethod
    def map(self, word2index: Dict[str, int]) -> np.ndarray:
        """Return a word embedding matrix

            Stack word vectors adhere to the order of index
            provided by the input word2index.

        """
        raise NotImplementedError
