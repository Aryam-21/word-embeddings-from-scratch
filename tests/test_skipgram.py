import math
import unittest

from src.model import SkipGram
from src.preprocessing import Preprocessor

TOL = 1e-5


def tiny_model():
    m = SkipGram(3, 2)                                # vocab: cats=0, eat=1, food=2
    m.E = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    m.U = [[0.0, 1.0, -1.0], [1.0, 0.0, 1.0]]
    return m


class TestSkipGram(unittest.TestCase):
    def test_context_window(self):
        pre = Preprocessor(window=1)
        sents = [["a", "b", "c", "d"]]
        pre.build_vocab(sents)
        w = pre.id2word
        got = [(w[c], w[x]) for c, x in pre.make_pairs(sents)]
        self.assertEqual(got, [("a", "b"), ("b", "a"), ("b", "c"),
                               ("c", "b"), ("c", "d"), ("d", "c")])

    def test_forward(self):
        h, s, p, loss = tiny_model().forward(0, 1)
        for x, y in zip(s, [0, 1, -1]):
            self.assertAlmostEqual(x, y, delta=TOL)

    def test_probabilities_and_loss(self):
        _, _, p, loss = tiny_model().forward(0, 1)
        for x, y in zip(p, [0.244728, 0.665241, 0.090031]):
            self.assertAlmostEqual(x, y, delta=TOL)
        self.assertAlmostEqual(sum(p), 1.0, delta=TOL)
        self.assertAlmostEqual(loss, 0.407606, delta=TOL)

    def test_gradients(self):
        gU, gh, _ = tiny_model().gradients(0, 1)
        for x, y in zip(gU[0], [0.244728, -0.334759, 0.090031]):
            self.assertAlmostEqual(x, y, delta=TOL)
        self.assertEqual(gU[1], [0.0, 0.0, 0.0])
        for x, y in zip(gh, [-0.424790, 0.334759]):
            self.assertAlmostEqual(x, y, delta=TOL)

    def test_one_update(self):
        m = tiny_model()
        m.step(0, 1, 0.1)
        for x, y in zip(m.E[0], [1.042479, -0.033476]):
            self.assertAlmostEqual(x, y, delta=TOL)
        self.assertEqual(m.E[1], [0.0, 1.0])          # other rows unchanged
        self.assertEqual(m.E[2], [1.0, 1.0])
        self.assertAlmostEqual(m.pair_loss(0, 1), 0.361859, delta=TOL)

    def test_extreme_scores(self):
        _, loss = SkipGram.softmax_loss([1000, 0], 1)
        self.assertAlmostEqual(loss, 1000, delta=TOL)

    def test_shift_invariance(self):
        p1, l1 = SkipGram.softmax_loss([0, 1, -1], 1)
        p2, l2 = SkipGram.softmax_loss([100, 101, 99], 1)
        self.assertAlmostEqual(l1, l2, delta=TOL)
        for a, b in zip(p1, p2):
            self.assertAlmostEqual(a, b, delta=TOL)


if __name__ == "__main__":
    unittest.main()