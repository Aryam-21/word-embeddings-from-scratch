# word-embeddings-from-scratch

A skip-gram neural network written in pure Python (lists, loops, `math`, `random`)
that learns word embeddings for **Tigrinya**. The forward pass, stable softmax,
loss, gradients and SGD are all implemented by hand. No NumPy, no automatic
differentiation, no neural-network libraries and no pretrained embeddings.

## Project structure

```
word-embeddings-from-scratch/
├── README.md
├── data/raw/tigrinya_corpus.txt     corpus, one sentence per line
├── models/tigrinya.json             saved vocabulary, E, U and settings
├── src/
│   ├── __init__.py
│   ├── preprocessing.py             Preprocessor: cleaning, vocabulary, training pairs
│   ├── model.py                     SkipGram: forward, loss, gradients, SGD step, save/load
│   ├── training.py                  Trainer: shuffled SGD loop, prints J per epoch
│   └── embeddings.py                Embeddings: cosine similarity, nearest neighbors
├── tests/test_skipgram.py           the required correctness checks
└── word_embeddings_tigrinya.ipynb   experiment and results
```

## How to run

Python 3 standard library only. No installation needed.

```bash
python -m unittest tests.test_skipgram     # run from the project root
jupyter notebook word_embeddings_tigrinya.ipynb
```

Run the notebook from the top. Restart the kernel after editing any file in `src/`.

## Corpus and preprocessing

- About 200 short Tigrinya sentences (see `data/raw/tigrinya_corpus.txt`).
- Sentences are split on the Ethiopic full stop (`።`) and newlines.
- Ethiopic and common ASCII punctuation is removed, then text is split on whitespace.
- The vocabulary is sorted alphabetically (`min_count` is a setting).
- Training pairs are (center, context) within a window, and **never cross sentence boundaries**.

## Model

Row-vector notation, with `V` the vocabulary size and `d` the embedding dimension.
`E` is V×d and `U` is d×V. Both start with small random values from a recorded seed.

| Step | Equation |
|---|---|
| Lookup | `h = E[i]` |
| Scores | `s = hU` |
| Stable softmax | `m = max(s)`, `a[j] = exp(s[j] − m)`, `p[j] = a[j] / sum(a)` |
| Loss | `L = (m − s[t]) + ln(sum(a))` |
| Score error | `e = p − onehot(t)` |
| Gradients | `grad_U[k,j] = h[k]·e[j]`, `grad_h[k] = sum_j U[k,j]·e[j]` |
| Update | `U ← U − lr·grad_U`, `E[i] ← E[i] − lr·grad_h` |

Both gradients are computed with the old weights before either matrix is updated,
and only the selected row of `E` changes. Pairs are shuffled every epoch.
The dataset loss `J` is the mean of `L` over all pairs, evaluated at fixed weights.

## Settings

| Setting | Value |
|---|---|
| Seed | 42 |
| Embedding dimension d | 10 |
| Context window | 3 |
| Learning rate | 0.05 |
| Epochs | 50 |
| min_count | 1 |


## Correctness checks

`tests/test_skipgram.py` runs these checks with tolerance 1e-5, using the tiny
cats/eat/food example:

1. Context window pairs, with no cross-sentence pairs
2. Forward pass scores `[0, 1, −1]`
3. Probabilities (sum to 1) and loss `L ≈ 0.407606`
4. Gradients `grad_U` and `grad_h`
5. One SGD update: new `E[cats]`, recomputed loss `≈ 0.361859`, other rows unchanged
6. Extreme scores `[1000, 0]`: stable loss `≈ 1000`
7. Shift invariance: adding 100 to every score changes nothing

## Results

Mean training loss J, same pairs, weights frozen during evaluation:

| Stage | J |
|---|---|
| Before training | 6.2064 |
| After epoch 15 | 3.1650 |

The starting value is close to `ln(V)`, the loss of a uniform guess over the
vocabulary. J falls every epoch, so training is working.

```
word -> three nearest vectors (cosine similarity)
```

## Limitations

- The corpus is tiny and Tigrinya is morphologically rich, so many words appear
  only once and their vectors are poorly trained. Nearest neighbors are expected to be weak.
- Lower training loss alone does not establish semantic quality.

## Optional extras

- Save and reload `E`, `U`, the vocabulary and the settings as JSON (`SkipGram.save` / `SkipGram.load`)
- Numerical gradient check with ε = 1e-5
- Comparing nearest neighbors after changing only the window or only d