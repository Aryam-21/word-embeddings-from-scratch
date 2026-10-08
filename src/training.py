import random


class Trainer:
    """SGD training loop; reports J before training and after every epoch."""

    def __init__(self, model, lr=0.05, epochs=10, seed=42):
        self.model, self.lr, self.epochs = model, lr, epochs
        self.rnd = random.Random(seed)
        self.history = []

    def fit(self, pairs):
        pairs = list(pairs)
        self.history = [self.model.dataset_loss(pairs)]
        print(f"before training : J = {self.history[0]:.4f}")
        for epoch in range(1, self.epochs + 1):
            self.rnd.shuffle(pairs)                   # shuffle every epoch
            for i, t in pairs:
                self.model.step(i, t, self.lr)
            self.history.append(self.model.dataset_loss(pairs))
            print(f"epoch {epoch:>3}       : J = {self.history[-1]:.4f}")
        return self.history