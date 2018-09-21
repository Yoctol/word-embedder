from pathlib import Path

from dotenv import load_dotenv

from .library import Library
from .keyed_vectors_light import KeyedVectorsLight


load_dotenv()

path = Path('.')
ROOT_DIR = path.parent

lib = Library()
lib.register(
    'ChineseFastTextLight',
    KeyedVectorsLight(path=str(ROOT_DIR.joinpath('data/ft.zh.300.vec').resolve())),
)
