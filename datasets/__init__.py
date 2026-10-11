from .register import DATASETS
from .rock_data import *

__all__ = ['DATASETS', *DATASETS.registry_names]