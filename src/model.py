import json
import math
import random


class SkipGram:
    """Skip-gram with full softmax. E is V x d, U is d x V (row-vector notation)."""

    def __init__(self, vocab_size, dim, seed=42):
        self.V, self.d, self.seed = vocab_size, dim, seed
        rnd = random.Random(seed)
        self.E = [[rnd.uniform(-0.1, 0.1) for _ in range(dim)] for _ in range(vocab_size)]
        self.U = [[rnd.uniform(-0.1, 0.1) for _ in range(vocab_size)] for _ in range(dim)]

    # ---------- forward ----------
    def scores(self, h):
        return [sum(h[k] * self.U[k][j] for k in range(self.d)) for j in range(self.V)]

    @staticmethod
    def softmax_loss(s, t):
        """Stable probabilities and loss: L = (m - s[t]) + ln(sum(a))."""
        m = max(s)
        a = [math.exp(x - m) for x in s]
        total = sum(a)
        p = [x / total for x in a]
        return p, (m - s[t]) + math.log(total)

    def forward(self, i, t):
        h = list(self.E[i])
        s = self.scores(h)
        p, loss = self.softmax_loss(s, t)
        return h, s, p, loss

    # ---------- backward ----------
    def gradients(self, i, t):
        h, s, p, loss = self.forward(i, t)
        e = list(p)
        e[t] -= 1.0                                   # e = p - onehot(t)
        grad_U = [[h[k] * e[j] for j in range(self.V)] for k in range(self.d)]
        grad_h = [sum(self.U[k][j] * e[j] for j in range(self.V)) for k in range(self.d)]
        return grad_U, grad_h, loss

    def step(self, i, t, lr):
        """One SGD update. Both gradients use the OLD weights."""
        grad_U, grad_h, loss = self.gradients(i, t)
        for k in range(self.d):
            for j in range(self.V):
                self.U[k][j] -= lr * grad_U[k][j]
            self.E[i][k] -= lr * grad_h[k]            # only row i of E changes
        return loss

    # ---------- evaluation ----------
    def pair_loss(self, i, t):
        return self.forward(i, t)[3]

    def dataset_loss(self, pairs):
        """J = mean loss at fixed weights (no updates)."""
        return sum(self.pair_loss(i, t) for i, t in pairs) / len(pairs)

    # ---------- save / load ----------
    def save(self, path, vocab, meta=None, digits=8):
        """Save vocab, E, U and training settings (meta) as readable JSON."""
        info = {"seed": self.seed, "vocab_size": self.V, "dim": self.d}
        info.update(meta or {})                       # e.g. window, lr, epochs, loss history
        rnd = lambda M: [[round(x, digits) for x in row] for row in M]
        with open(path, "w", encoding="utf-8") as f:
            f.write('{\n  "meta": ' + json.dumps(info, ensure_ascii=False))
            f.write(',\n  "vocab": ' + json.dumps(vocab, ensure_ascii=False))
            for name, M in (("E", rnd(self.E)), ("U", rnd(self.U))):
                rows = ",\n    ".join(json.dumps(row) for row in M)   # one row per line
                f.write(',\n  "%s": [\n    %s\n  ]' % (name, rows))
            f.write("\n}\n")

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta", {"seed": data.get("seed", 42)})   # old files: top-level seed
        model = cls(len(data["vocab"]), len(data["E"][0]), meta.get("seed", 42))
        model.E, model.U, model.meta = data["E"], data["U"], meta
        return model, data["vocab"]