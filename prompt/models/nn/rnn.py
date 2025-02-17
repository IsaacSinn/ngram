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

class RNN(pt.nn.Module):
    """
    In this file you should implement a generic RNN language
    model with the class RNN. Your class should follow the same api as models/nn/unigram.Unigram.
    Your model should contain an instance of the RNNCell class in Pytorch.
    """
    def __init__(self: RNNType, data: Sequence[Sequence[str]], saved_model_path: str = None, num_epochs: int = 2) -> None:
        super().__init__()

        self.vocab = Vocab()

        for line in data: 
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # |vocab| -> 64 -> |vocab| 
        
        self.rnn = pt.nn.RNNCell(len(self.vocab), 64)
        self.fc = pt.nn.Linear(64, len(self.vocab))

        self.error = pt.nn.CrossEntropyLoss()
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)
    
        if saved_model_path is None:
            o = pt.optim.SGD(self.rnn.parameters(), lr=1e-3)

            for epoch in range(num_epochs):

                random.shuffle(data)
                train_chars = 0

                for line in tqdm(data, desc=f"epoch {epoch}"):

                    self.optimizer.zero_grad()
                    loss = 0.

                    q = self.start()

                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]): # skip BOS
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))
                        
                        # ground truth is a vector of length |vocab| with a 1 at the index of the correct character
                        ground_truth = pt.tensor([self.vocab.numberize(c_out)])

                        loss += self.error(p, ground_truth)
                    
                    loss.backward()

                    self.optimizer.step()

                    # gradient clip
                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                # save model parameters
                pt.save(self.state_dict(), f"./rnn_{epoch}.model")
        else:
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

    
    def start(self: RNNType) -> StateType:
        return pt.zeros(1, 64)

    def forward(self: RNNType, input_word: int, hidden) -> pt.Tensor:
        one_hot = pt.zeros(1, len(self.vocab))
        one_hot[0, input_word] = 1

        hidden = self.rnn(one_hot, hidden)
        
        output_logits = self.fc(hidden)
        
        # output_logits: [batch, vocab_size]
        return output_logits, hidden


    def step(self: RNNType, state: StateType, x: int) -> Tuple[StateType, Mapping[str, float]]:
        logits, state = self.forward(x, state)
        # Removed softmax so that logits (raw scores) are returned for training loss.
        return state, pt.log_softmax(logits, dim=1)