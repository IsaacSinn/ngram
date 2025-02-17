# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
from pprint import pprint
import argparse as ap
import numpy as np
import os
import sys
from tqdm import tqdm  # new import for progress bar
import pickle  # new import for saving model
from models.ngram.ngram_vanilla import Ngram as Ngram_vanilla  # new import for Ngram_vanilla

# make sure the directory that contains this file is in sys.path
_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_

# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
from models.ngram.ngram import Ngram
from vocab import START_TOKEN, END_TOKEN

def train_ngram(n: int) -> Ngram:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    return Ngram(n, train_data)

# New function to train a vanilla n-gram model.
def train_ngram_vanilla(n: int) -> Ngram_vanilla:
    train_data: Sequence[Sequence[str]] = load_chars_from_file("./data/large")
    return Ngram_vanilla(n, train_data)

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

def train_and_save() -> None:
    m: Ngram = train_ngram(5)
    print("Training complete: ", len(m.context_counts))
    
    # Save the trained model
    with open("ngram_model.pkl", "wb") as f:  # model saved in current directory
        pickle.dump(m, f)
    print("Model saved as ngram_model.pkl")
    
    num_correct, total = dev_ngram(m)
    print("Dev accuracy:", num_correct / total)

# New function to train and save the vanilla model.
def train_and_save_vanilla() -> None:
    m_v = train_ngram_vanilla(5)
    print("Vanilla training complete: ", len(m_v.context_counts))
    
    with open("ngram_vanilla_model.pkl", "wb") as f:
        pickle.dump(m_v, f)
    print("Ngram_vanilla model saved as ngram_vanilla_model.pkl")

def open_and_test() -> None:
    with open("ngram_model.pkl", "rb") as f:
        m: Ngram = pickle.load(f)
    print("Model loaded from ngram_model.pkl")
    
    num_correct, total = dev_ngram(m)
    print("Test accuracy:", num_correct / total)

if __name__ == "__main__":

    train_and_save()
    train_and_save_vanilla()

    # open_and_test()
