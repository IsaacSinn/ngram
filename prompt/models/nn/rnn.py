# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import collections, math, random, sys
import os
import sys
import torch as pt
from tqdm import tqdm

_cd_: str = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..", "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

RNNType: Type = Type["RNN"]

class RNN():
    def __init__(self: RNNType, data: Sequence[Sequence[str]], saved_model_path: str = None, num_epochs: int = 2) -> None:
        super().__init__()

        self.vocab = Vocab()