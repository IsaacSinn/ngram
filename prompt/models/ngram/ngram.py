from typing import Type, Tuple, Dict, Sequence, Mapping
import collections
import numpy as np
from tqdm import tqdm  # new import for progress bar

# PYTHON PROJECT IMPORTS
from lm import LM, StateType
from vocab import Vocab, START_TOKEN, END_TOKEN

# Types declared in this module
NgramType: Type = Type["Ngram"]

class Ngram(LM):
    '''
    An n-gram language model using Absolute Discounting with Recursive Backoff.
    '''
    def __init__(self, n: int, data: Sequence[Sequence[str]]) -> None:
        self.n: int = n
        self.vocab: Vocab = Vocab()
        self.context_counts: collections.Counter = collections.defaultdict(collections.Counter)
        self.total_counts: collections.Counter = collections.Counter()

        # store n minus 1 gram for absolute discounting
        self.n_minus_1_gram_counts: collections.Counter = collections.defaultdict(collections.Counter)
        self.total_counts_n_minus_1: collections.Counter = collections.Counter()

        self.discount: float = 0.6  # d

        self.debug: bool = False

        # First pass
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)

        # Second pass
        for line in tqdm(data, desc="Counting"):
            tokens = [START_TOKEN] * (self.n - 1) + list(line) + [END_TOKEN]
            for i in range(len(tokens) - self.n + 1):
                context = tuple(self.vocab.numberize(t) for t in tokens[i: i + self.n - 1])
                current = self.vocab.numberize(tokens[i + self.n - 1])
                self.context_counts[context][current] += 1
                self.total_counts[context] += 1

                self.n_minus_1_gram_counts[context[1:]][current] += 1
                self.total_counts_n_minus_1[context[1:]] += 1

        vocab_size: int = len(self.vocab)

        self.logprob: Dict[Tuple[int, ...], np.ndarray] = {}

        # Initialize logprob arrays for seen contexts with NaN to mark "not computed"
        for context in self.context_counts.keys():
            self.logprob[context] = np.full(vocab_size, np.nan, dtype=float)

        if self.debug:
            print("vocab size", vocab_size)

        # Precompute probabilities for each observed context.
        for context in tqdm(list(self.context_counts.keys()), desc="computing logprob"):
        # for context in self.context_counts.keys():
            for w_idx in range(vocab_size):

                if self.debug:
                    # print context and the word
                    print(f"context: {[self.vocab.denumberize(i) for i in context]}, word: {self.vocab.denumberize(w_idx)}")
                    # print the count of the word in the context
                    print(f"count({self.vocab.denumberize(w_idx)}|{[self.vocab.denumberize(i) for i in context]}) = {self.context_counts[context][w_idx]}")
                    # print total count of the context
                    print(f"count({[self.vocab.denumberize(i) for i in context]}) = {self.total_counts[context]}")

                prob = self.get_gram_prob(context, w_idx)
                self.logprob[context][w_idx] = np.log(prob) if prob > 0 else -np.inf

                if self.debug:
                    # print the log probability of the word in the context
                    print(f"final logprob({self.vocab.denumberize(w_idx)}|{[self.vocab.denumberize(i) for i in context]}) = {self.logprob[context][w_idx]}")
                    print("--------------")

            if self.debug:
                print("=====================================")

    def get_gram_prob(self, context: Tuple[int, ...], w_idx: int) -> float:
        """
        compute gram prob smoothed with absolute discounting
        Pr[a|u] = (max(0, c(a|u)-d)/total(u)) + (backoff_weight * Pr[a|u~])
        where u~ is the n-1 gram context obtained by removing the first element.
        """
        
        # Base case: n -1 gram, or n gram context not seen before.
        if len(context) == self.n - 2 or context not in self.context_counts:
            if self.debug:
                print("1")

            if context not in self.n_minus_1_gram_counts:
                return 0.0
            else:
                prob = self.n_minus_1_gram_counts[context][w_idx] / self.total_counts_n_minus_1[context]
                return prob

        counts = self.context_counts[context]
        total = self.total_counts[context]
        n1plus = sum(1 for count in counts.values() if count > 0)

        # Probability mass assigned to the observed count for w_idx.
        prob_observed = max(0, counts[w_idx] - self.discount) / total if total > 0 else 0
        if self.debug:
            print("prob observed for ", self.vocab.denumberize(w_idx), "is", prob_observed)

        # Backoff weight: redistributed probability mass.
        backoff_weight = (self.discount + n1plus / total) if total > 0 else 0
        if self.debug:
            print("baxkoff weight for ", self.vocab.denumberize(w_idx), "is", backoff_weight)
        backoff_prob = self.get_gram_prob(context[1:], w_idx)
        if self.debug:
            print("backoff prob for ", self.vocab.denumberize(w_idx), "is", backoff_prob)
        prob = prob_observed + backoff_weight * backoff_prob
        if self.debug:
            print("prob (no log), for ", self.vocab.denumberize(w_idx), "is", prob)

        return prob

    def start(self: NgramType) -> StateType:
        """Return the language model's start state."""
        state: StateType = np.array([self.vocab.numberize(START_TOKEN)] * (self.n - 1))
        return state

    def step(self: NgramType, q: StateType, w_idx: int) -> Tuple[StateType, Mapping[str, float]]:
        """Compute one step of the language model."""
        q = np.append(q[1:], w_idx)
        context = tuple(q)

        logprobs = np.full(len(self.vocab), -np.inf)

        if context not in self.logprob:
            uniform_logprob = np.log(1.0 / (len(self.vocab) - 1))
            logprobs.fill(uniform_logprob)
        else:
            for word_idx, lp in enumerate(self.logprob.get(context, [])):
                logprobs[word_idx] = lp

        start_token_idx = self.vocab.numberize(START_TOKEN)
        logprobs[start_token_idx] = -np.inf

        # Normalize
        max_logprob = np.max(logprobs)
        logprobs -= max_logprob
        logprobs -= np.log(np.sum(np.exp(logprobs)))

        return (q, logprobs)
