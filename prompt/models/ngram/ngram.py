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
    def __init__(self, n: int, data: Sequence[Sequence[str]], discount: float = 0.1) -> None:
        self.n: int = n
        self.vocab: Vocab = Vocab()
        self.context_counts: collections.Counter = collections.defaultdict(collections.Counter)
        self.total_counts: collections.Counter = collections.Counter()
        self.unigram_counts: collections.Counter = collections.Counter()

        self.discount: float = discount  # d

        # First pass
        for line in data:
            for w in list(line) + [END_TOKEN]:
                self.vocab.add(w)
                # Count unigrams (using the numberized form)
                self.unigram_counts[self.vocab.numberize(w)] += 1

        # Second pass
        for line in tqdm(data, desc="Training"):  # added progress bar
            tokens = [START_TOKEN] * (self.n - 1) + list(line) + [END_TOKEN]
            for i in range(len(tokens) - self.n + 1):
                context = tuple(self.vocab.numberize(t) for t in tokens[i: i + self.n - 1])
                current = self.vocab.numberize(tokens[i + self.n - 1])
                self.context_counts[context][current] += 1
                self.total_counts[context] += 1

        vocab_size: int = len(self.vocab)
        self.logprob: Dict[Tuple[int, ...], np.ndarray] = {}

        # Initialize logprob arrays for seen contexts with NaN to mark "not computed"
        for context in self.context_counts.keys():
            self.logprob[context] = np.full(vocab_size, np.nan, dtype=float)

        # Precompute probabilities for each observed context.
        for context in tqdm(list(self.context_counts.keys()), desc="Precomputing probabilities"):
            for w_idx in range(vocab_size):
                prob = self.get_gram_prob(context, w_idx)
                self.logprob[context][w_idx] = np.log(prob) if prob > 0 else -np.inf

    def get_gram_prob(self, context: Tuple[int, ...], w_idx: int) -> float:
        """
        Recursively computes the smoothed probability for word index w_idx given context,
        using Absolute Discounting with backoff.
        """
        # Base case: Unigram probability (empty context)
        if len(context) == 0:
            total_unigram_count = sum(self.unigram_counts.values())
            if total_unigram_count == 0:
                return 1 / len(self.vocab)
            return self.unigram_counts[w_idx] / total_unigram_count

        # If the context was never seen in training, back off.
        if context not in self.context_counts:
            return self.get_gram_prob(context[1:], w_idx)

        # If probability already computed, return the cached value.
        # (We check if the log probability for this word is not NaN.)
        if context in self.logprob and not np.isnan(self.logprob[context][w_idx]):
            return np.exp(self.logprob[context][w_idx])

        counts = self.context_counts[context]
        total = self.total_counts[context]
        n1plus = sum(1 for count in counts.values() if count > 0)

        # Probability mass assigned to the observed count for w_idx.
        prob_observed = max(0, counts[w_idx] - self.discount) / total if total > 0 else 0

        # Backoff weight: redistributed probability mass.
        backoff_weight = (self.discount + n1plus / total) if total > 0 else 0
        backoff_prob = self.get_gram_prob(context[1:], w_idx)
        prob = prob_observed + backoff_weight * backoff_prob


        self.logprob[context][w_idx] = np.log(prob) if prob > 0 else -np.inf

        return prob

    def start(self: NgramType) -> StateType:
        """Return the language model's start state."""
        return tuple(self.vocab.numberize(START_TOKEN) for _ in range(self.n - 1))

    def step(self: NgramType, q: StateType, w_idx: int) -> Tuple[StateType, Mapping[str, float]]:
        """Compute one step of the language model."""
        if self.n == 1:
            return ()
        else:
            new_context = (*q[1:], w_idx)

        if new_context not in self.logprob:
            return new_context, np.full(len(self.vocab), -np.inf)
        else:
            return new_context, self.logprob[new_context]
