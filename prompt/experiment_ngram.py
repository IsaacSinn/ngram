# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
from pprint import pprint
import argparse as ap
import numpy as np
import os
import sys
from tqdm import tqdm  # new import for progress bar

# make sure the directory that contains this file is in sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_

# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.ngram.ngram import Ngram
from vocab import START_TOKEN, END_TOKEN

def train_ngram() -> Ngram:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    return Ngram(5, train_data)

def dev_ngram(m: Ngram) -> Tuple[int, int]:
    dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/dev")

    num_correct: int = 0
    total: int = 0
    # Wrap dev_data with tqdm for progress tracking
    for dev_line in tqdm(dev_data, desc='Testing on dev data'):
        q = m.start()

        for c_input, c_actual in zip([START_TOKEN] + dev_line, # read in string w/ <BOS> prepended
                                      dev_line + [END_TOKEN]): # check against string incl. <EOS>
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))

            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total

def test_ngram(m: Ngram) -> Tuple[int, int]:
    test_data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")
    num_correct: int = 0
    total: int = 0
    for line in tqdm(test_data, desc="Testing on test data"):
        q = m.start()
        for c_input, c_actual in zip([START_TOKEN] + line,  # prepend <BOS>
                                      line + [END_TOKEN]):     # append <EOS>
            q, p = m.step(q, m.vocab.numberize(c_input))
            c_predicted = m.vocab.denumberize(np.argmax(p))
            num_correct += int(c_predicted == c_actual)
            total += 1
    return num_correct, total

def main() -> None:
    m: Ngram = train_ngram()

    num_correct, total = test_ngram(m)

    print(num_correct / total)

if __name__ == "__main__":
    main()
