import math


class Embeddings:
    """Query trained vectors (rows of E)."""

    def __init__(self, E, word2id, id2word):
        self.E, self.word2id, self.id2word = E, word2id, id2word

    @staticmethod
    def cosine(a, b):
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:                        # zero-norm: avoid division by 0
            return 0.0
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    def nearest(self, word, k=3):
        if word not in self.word2id:                  # unknown word
            return []
        i = self.word2id[word]
        scored = [(self.cosine(self.E[i], self.E[j]), self.id2word[j])
                  for j in range(len(self.E)) if j != i]
        scored.sort(reverse=True)
        return [(w, round(s, 4)) for s, w in scored[:k]]