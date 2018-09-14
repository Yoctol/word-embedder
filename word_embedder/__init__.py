from .library import Library
from .fasttext import FastText

lib = Library()
lib.register('ChineseFastText', FastText(path='OHOH'))
