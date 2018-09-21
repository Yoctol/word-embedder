from pathlib import Path

from dotenv import load_dotenv

from .library import Library
from .keyed_vectors_light import KeyVectorsLight


load_dotenv()

path = Path('.')
ROOT_DIR = path.parent

lib = Library()
lib.register(
    'ChineseFastTextLight',
    KeyVectorsLight(path=str(ROOT_DIR.joinpath('data/ft.zh.300.vec').resolve())),
)
