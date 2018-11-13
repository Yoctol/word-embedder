from typing import List
from collections import OrderedDict


class Library:

    def __init__(self):
        self.library = OrderedDict()

    def register(
            self,
            name: str,
            embedder: object,
        ) -> None:
        if name not in self.library:
            self.library[name] = embedder
        else:
            raise KeyError('Please use anther name for new embedder')

    def list_all(self) -> List[str]:
        return list(self.library.keys())

    def __getitem__(self, name: str):
        if name not in self.library:
            KeyError('Embedder [{}] is not found'.format(name))
        else:
            self.library[name].build()
            return self.library[name]
