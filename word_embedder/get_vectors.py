import warnings
from typing import List

import numpy as np

from .embedders import Embedder


def get_vectors(
        embedder: Embedder,
        words: List[str] = None,
        seed: int = 2018,
        dtype: str = 'float32',
    ) -> np.ndarray:

    if words is None:
        words = embedder.vocab

    words = remove_duplicated_words(words)

    rand_state = np.random.RandomState(seed)
    embedding = rand_state.rand(len(words), embedder.n_dim).astype(dtype)

    for i, word in enumerate(words):
        ind = embedder.get_index(word)
        if ind != -1:
            embedding[i, :] = embedder[ind]
        else:
            embedding[i, :] = embedding[i, :] / np.linalg.norm(embedding[i, :])
    return embedding


def remove_duplicated_words(words: List[str]) -> List[str]:

    if len(words) != len(set(words)):
        duplicate = []
        visited = {word: False for word in set(words)}
        unique_words = []
        for word in words:
            if visited[word]:
                duplicate.append(word)
            else:
                visited[word] = True
                unique_words.append(word)
        warnings.warn(
            f"The input words are not unique. Duplicated elements are {duplicate}.",
            RuntimeWarning,
        )
        return unique_words
    else:
        return words
