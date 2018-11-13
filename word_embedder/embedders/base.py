from abc import ABC, abstractmethod, abstractproperty
from typing import List

import numpy as np


class Embedder(ABC):

    @abstractmethod
    def __getitem__(self, key) -> np.ndarray:
        """Get a word vector

            If key is an int, return vector by index.
            If key is a string, return vector by word.

        """
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @abstractproperty
    def n_vocab(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def n_dim(self) -> int:
        raise NotImplementedError

    @abstractproperty
    def vocab(self) -> List[str]:
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
