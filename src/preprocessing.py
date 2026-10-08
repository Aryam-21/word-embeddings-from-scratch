import re


class Preprocessor:
    """Load text, build the vocabulary, create (center, context) pairs."""

    def __init__(self, window=2, min_count=1):
        self.window = window
        self.min_count = min_count
        self.word2id = {}
        self.id2word = []

    @staticmethod
    def tokenize(text):
        # remove Ethiopic punctuation (U+1360-1368) and common ASCII punctuation
        text = re.sub(r"[\u1360-\u1368.,;:!?\"'()\[\]\u201c\u201d\u2018\u2019-]", " ", text)
        return text.split()

    def load_sentences(self, path):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        # sentences end with the Ethiopic full stop (U+1362) or a newline
        chunks = re.split(r"[\u1362\n]", text)
        sentences = [self.tokenize(c) for c in chunks]
        return [s for s in sentences if s]

    def build_vocab(self, sentences):
        counts = {}
        for sent in sentences:
            for w in sent:
                counts[w] = counts.get(w, 0) + 1
        self.id2word = sorted(w for w, c in counts.items() if c >= self.min_count)
        self.word2id = {w: i for i, w in enumerate(self.id2word)}

    def make_pairs(self, sentences):
        pairs = []
        for sent in sentences:  # pairs never cross sentence boundaries
            ids = [self.word2id[w] for w in sent if w in self.word2id]
            for pos, center in enumerate(ids):
                lo = max(0, pos - self.window)
                hi = min(len(ids), pos + self.window + 1)
                for j in range(lo, hi):
                    if j != pos:
                        pairs.append((center, ids[j]))
        return pairs