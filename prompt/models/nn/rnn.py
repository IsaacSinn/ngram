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
from data.charloader import load_chars_from_file

RNNType: Type = Type["RNN"]
LSTMType: Type = Type["LSTM"]

class RNN(pt.nn.Module):
    """
    """
    def __init__(self: RNNType, data: Sequence[Sequence[str]], saved_model_path: str = None, num_epochs: int = 1) -> None:
        super().__init__()

        self.vocab = Vocab()

        self.vocab.add(START_TOKEN)

        for line in data: 
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # |vocab| -> 64 -> |vocab| 
        
        self.rnn = pt.nn.RNNCell(len(self.vocab), 64)
        self.fc = pt.nn.Linear(64, len(self.vocab))

        self.error = pt.nn.CrossEntropyLoss()
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)
    
        if saved_model_path is None:
            for epoch in range(num_epochs):

                random.shuffle(data)
                train_chars = 0

                for line in tqdm(data, desc=f"epoch {epoch}"):

                    self.optimizer.zero_grad()
                    loss = 0.

                    q = self.start()

                    for c_in, c_out in zip([START_TOKEN] + line, line + [END_TOKEN]):
                        train_chars += 1
                        q, p = self.step(q, self.vocab.numberize(c_in))
                        
                        ground_truth = pt.tensor([self.vocab.numberize(c_out)])

                        loss += self.error(p, ground_truth)
                    
                    loss.backward()

                    self.optimizer.step()

                    pt.nn.utils.clip_grad_norm_(self.parameters(), 1.0)

                pt.save(self.state_dict(), f"./rnn_{epoch}.model")
        else:
            print(f"loading model from {saved_model_path}")
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

        logits = logits.squeeze(0)

        return state, pt.log_softmax(logits, dim=0)
    

class LSTM(pt.nn.Module):
    """
    """
    def __init__(self: LSTMType, data: Sequence[Sequence[str]], saved_model_path: str = None, num_epochs: int = 15) -> None:
        super().__init__()

        # self.device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
        self.device = pt.device("cpu")

        print(f"device: {self.device}")
        
        self.vocab = Vocab()
        for line in data: 
            for w in list(line):
                self.vocab.add(w)

        # print bos and eos
        print(self.vocab.denumberize(0))
        print(self.vocab.denumberize(1))
        print(self.vocab.denumberize(2))

        print(f"vocab size: {len(self.vocab)}")

        # |vocab| -> 64 -> |vocab|
        self.lstm = pt.nn.LSTMCell(len(self.vocab), 64)
        self.fc = pt.nn.Linear(64, len(self.vocab))

        self.error = pt.nn.CrossEntropyLoss()
        self.optimizer = pt.optim.Adam(self.parameters(), lr=1e-3)
        
        self.to(self.device)
    
        if saved_model_path is None:
        

            for epoch in range(num_epochs):
                random.shuffle(data)
                train_chars = 0
                for line in tqdm(data, desc=f"epoch {epoch}"):
                    self.optimizer.zero_grad()
                    loss = 0.
                    q = self.start()
                    for c_input, c_actual in zip([START_TOKEN] + line, line + [END_TOKEN]):
                        q, p = self.step(q, self.vocab.numberize(c_input))
                        loss += self.error(p, pt.tensor([self.vocab.numberize(c_actual)], device=self.device))
                    loss.backward()
                    self.optimizer.step()
                    train_chars += len(line)
                print(f"epoch {epoch} loss: {loss.item() / train_chars}")
                pt.save(self.state_dict(), f"./lstm_test_{epoch}.model")


                # test on test set
                dev_data: Sequence[Sequence[str]] = load_chars_from_file("./data/test")
                num_correct: int = 0
                total: int = 0
                for dev_line in dev_data:
                    q = self.start()
                    for c_input, c_actual in zip([START_TOKEN] + dev_line, dev_line + [END_TOKEN]):
                        q, p = self.step(q, self.vocab.numberize(c_input))
                        c_predicted = self.vocab.denumberize(p.argmax())
                        num_correct += int(c_predicted == c_actual)
                        total += 1
                print(f"epoch {epoch} dev accuracy: {num_correct / total}")
        else:
            print(f"loading model from {saved_model_path}")
            self.load_state_dict(pt.load(saved_model_path, weights_only=True))

            # print the layer sizes
            print(f"lstm weight_ih: {self.lstm.weight_ih.size()}")
        

    def start(self: LSTMType) -> StateType:

        return pt.zeros(1, 64, device=self.device), pt.zeros(1, 64, device=self.device)

    def forward(self: LSTMType, q: StateType, c: int) -> Tuple[StateType, pt.Tensor]:

        one_hot = pt.zeros(1, len(self.vocab), device=self.device)
        one_hot[0, c] = 1.
        new_hidden, new_cell = self.lstm(one_hot, q)
        logits = self.fc(new_hidden)
        return (new_hidden, new_cell), logits

    def step(self: LSTMType, q: StateType, c: int) -> Tuple[StateType, pt.Tensor]:
        new_states, logits = self.forward(q, c)

        logits = logits.squeeze(0)

        return new_states, pt.log_softmax(logits, dim=0)