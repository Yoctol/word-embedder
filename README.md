# Word Embedder

[![travis][travis-image]][travis-url]
[![pypi][pypi-image]][pypi-url]

[travis-image]: https://img.shields.io/travis/Yoctol/word-embedder.svg?style=flat
[travis-url]: https://travis-ci.org/Yoctol/word-embedder
[pypi-image]: https://img.shields.io/pypi/v/word-embedder.svg?style=flat
[pypi-url]: https://pypi.python.org/pypi/word-embedder

A wrapper for loading pretrained word embedding


## Getting Started

### Prerequisites

Python 3.6 and up

### Installing

```
$ pip install word-embedder
```

## Usage

### Lookup all existed embedders
```python

from word_embedder.embedders import lib

lib.list_all_embedders()  # returns a list of embedder name

```

### Use an existed embedder

1. load an embedder called OHOH

```python

from word_embedder.embedders import lib

name = 'OHOH'  # embedder name
embedder = lib[name]

```

2. extract a word vector

- (1) given a word 'juice' (str)

    ```python

    word = 'juice'
    embedder[word]  # returns the corresponding word vector

    # Note: if 'juice' is not in the vocabulary, 
    # OOVError would be raised.

    ```

- (2) given an index 3 (int)

    ```python

    index = 3
    embedder[index]  # returns the corresponding word vector

    # Note: if the index is out of range of vocabulary size,
    # OOVError would be raised.

    ```

## For Development

### Running the tests

1. Coding style checking
```
$ make lint
```

2. Running unittests
```
$ make test
```

