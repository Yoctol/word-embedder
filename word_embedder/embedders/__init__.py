from pathlib import Path

from dotenv import load_dotenv

from .library import Library

from .base import Embedder  # noqa
from .keyed_vectors import KeyedVectors   # noqa
from .keyed_vectors_light import KeyedVectorsLight   # noqa
from .keyed_vectors_on_disk import KeyedVectorsOnDisk  # noqa


load_dotenv()

ROOT_DIR = Path(__file__).parent

lib = Library()
lib.register(
    'ChineseFastTextLight',
    KeyedVectorsLight(path=str(ROOT_DIR.joinpath('data/ft.zh.300.vec').resolve())),
)
