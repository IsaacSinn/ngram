from typing import Type, Tuple, Dict
from collections.abc import Sequence, Mapping
import collections
import numpy as np

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN


# Types declared in this module
NgramType: Type = Type["Ngram"]


class Ngram(LM):
    '''
    A n-gram language model.

    For each a in /sigma, u in /sigma^{n-1} in train data, count number of times a comes after u.
    Then, for each a in /sigma, compute P(a | u) = C(u, a) / C(u)
    
    '''
    def __init__(self, n: int, data:Sequence[Sequence[str]]) -> None:

        self.n: int = n
        self.vocab: Vocab = Vocab()
        self.context_counts: collections.Counter = collections.defaultdict(collections.Counter)
        self.total_counts: collections.Counter = collections.Counter()

        # context_counts stores: {(context): {next_word: count}}
        # total_counts stores: {context: count}

        # ABSOLUTE DISCOUNTING
        self.discount: float = 0.1 # d
        self.number_seen_1: int = 0 # n_{1+}

        # first pass
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
        
        # second pass
        for line in data:
            tokens = [START_TOKEN] * (self.n - 1) + list(line) + [END_TOKEN]

            # [1,2,3,4,5] n = 2, then 5 - 2 + 1 = 4 iterations
            for i in range(len(tokens) - self.n + 1):
                
                context = [self.vocab.numberize(t) for t in tokens[i: i + self.n - 1]]
                context = tuple(context)

                current = self.vocab.numberize(tokens[i + self.n - 1])


                self.context_counts[context][current] += 1
                self.total_counts[context] += 1

                if not self.vocab.__contains__(tokens[i + self.n - 1]):
                    self.number_seen_1 += 1
    
        
        vocab_size: int = len(self.vocab) # |sigma|

        self.logprob: Dict[Tuple[int, ...], np.ndarray] = {}

        for context, counts in self.context_counts.items():

            self.logprob[context] = np.zeros(vocab_size, dtype=float)

            for w_idx in range(vocab_size):
                self.logprob[context][w_idx] = np.log(counts[w_idx] + self.logprob[context][w_idx]/self.total_counts[context]) if counts[w_idx] > 0 else -np.inf


        # print all the tokens from smallest id to largest id
        # for w_idx in range(vocab_size):
        #     print(w_idx, self.vocab.denumberize(w_idx))

    def start(self: NgramType) -> StateType:
        
        """Return the language model's start state. (A unigram model doesn't
        have a state, so it's just `None`."""
        
        state: StateType = np.array([self.vocab.numberize(START_TOKEN)] * (self.n - 1))
        return state

    def step(self: NgramType, q: StateType, w_idx: int) -> Tuple[StateType, Mapping[str, float]]:
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - w_idk: The most recently seen numberized token (int)

        Return: (r, pb), where
        - r: The state of the model after reading 'w_idk'
        - pb: The log-probability distribution over the next token (after reading 'w_idx')
        """


        q = np.append(q[1:], w_idx)
        context = tuple(q)

        logprobs = np.full(len(self.vocab), -np.inf)

        # check if logprob for this context is empty
        if context not in self.logprob:
            uniform_logprob = np.log(1.0 / (len(self.vocab) - 1))
            logprobs.fill(uniform_logprob)
        else:
            for word_idx, lp in enumerate(self.logprob.get(context, [])):
                logprobs[word_idx] = lp

        start_token_idx = self.vocab.numberize(START_TOKEN)
        logprobs[start_token_idx] = -np.inf

        # Normalize the log probabilities
        max_logprob = np.max(logprobs)
        logprobs -= max_logprob
        logprobs -= np.log(np.sum(np.exp(logprobs)))

        return (q, logprobs)

                
                






