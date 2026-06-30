"""Numpy-only tests for the GeST core (tokenizer / SPE / serialization / mask).

Runnable without torch:  python -m gest.tests.test_numpy_core
Validates the GeST-specific novel bits the paper specifies, since the torch
transformer can only run on the farm.
"""
from __future__ import annotations

import numpy as np


def test_tokenizer():
    from gest.tokenizer import MetaCellVocab

    rng = np.random.default_rng(0)
    # 3 well-separated expression blobs (counts), 60 genes
    T, per = 60, 80
    centers = rng.integers(1, 20, size=(3, T)).astype(float)
    X = np.vstack([
        np.clip(centers[c] + rng.normal(0, 0.5, size=(per, T)), 0, None)
        for c in range(3)
    ]).astype(np.float32)
    true_blob = np.repeat(np.arange(3), per)

    voc = MetaCellVocab.fit(X, n_meta=12, level_sizes=[6, 3], n_pca=10, seed=0)
    assert voc.n_meta == 12 and voc.n_genes == T
    assert voc.level_sizes == [12, 6, 3]
    # 4 levels? paper uses 4; default level_sizes here gives 3 -> we passed 2 coarse
    assert len(voc.level_labels) == 3
    for i, sz in enumerate(voc.level_sizes):
        lab = voc.level_labels[i]
        assert lab.shape == (12,)
        assert lab.min() >= 0 and lab.max() < sz

    tok = voc.tokenize(X)
    assert tok.shape == (X.shape[0],) and tok.min() >= 0 and tok.max() < 12
    from collections import Counter
    # META-CELL purity: a fine meta cell shouldn't mix two well-separated blobs.
    meta_purity = []
    for m in np.unique(tok):
        blobs = true_blob[tok == m]
        meta_purity.append(Counter(blobs).most_common(1)[0][1] / blobs.size)
    assert min(meta_purity) > 0.95, meta_purity
    # COARSEST level (3 labels) should recover the 3 blobs: each blob -> one
    # dominant coarse label, and the blobs map to 3 distinct labels.
    labs = voc.labels_all_levels(X)
    coarse = labs[-1]                               # level size 3
    blob_to_label, label_purity = {}, []
    for c in range(3):
        sub = coarse[true_blob == c]
        lbl, cnt = Counter(sub).most_common(1)[0]
        blob_to_label[c] = lbl
        label_purity.append(cnt / sub.size)
    assert min(label_purity) > 0.95, label_purity
    assert len(set(blob_to_label.values())) == 3    # 3 blobs -> 3 distinct labels
    # token_expr nonneg + right shape; level 0 == tokenize
    te = voc.token_expr(X)
    assert te.shape == X.shape and te.min() >= 0
    assert len(labs) == 3 and np.array_equal(labs[0], tok)
    print(f"ok  tokenizer: meta-purity>={min(meta_purity):.3f}, "
          f"coarse recovers blobs (purity>={min(label_purity):.3f}), "
          f"level_sizes={voc.level_sizes}")


def test_level_aggregation():
    """The hierarchical-prob aggregation (Eq. 8) must keep probabilities summing
    to 1 at every level (np mirror of model.hierarchical_loss's index_add)."""
    from gest.tokenizer import MetaCellVocab
    rng = np.random.default_rng(1)
    X = np.clip(rng.normal(5, 2, size=(120, 40)), 0, None).astype(np.float32)
    voc = MetaCellVocab.fit(X, n_meta=16, level_sizes=[8, 4], n_pca=8, seed=1)
    p = rng.random(voc.n_meta); p = p / p.sum()      # a meta-cell distribution
    for i, sz in enumerate(voc.level_sizes):
        agg = np.zeros(sz)
        np.add.at(agg, voc.level_labels[i], p)       # sum_c [l_i(c)==k] p(c)
        assert abs(agg.sum() - 1.0) < 1e-6, (i, agg.sum())
    print("ok  level aggregation preserves probability mass")


def test_spe():
    from gest.positional import spe, spe_dim
    assert spe_dim(130) == 128 and spe_dim(128) == 128 and spe_dim(3) == 4
    coords = np.array([[0, 0], [1, 0], [0, 1], [5, -3]], dtype=float)
    e = spe(coords, 64)
    assert e.shape == (4, 64)
    assert np.all(np.abs(e) <= 1.0 + 1e-6)           # sin/cos bounded
    assert np.allclose(spe(coords, 64), e)           # deterministic
    # distinct coords -> distinct embeddings
    assert not np.allclose(e[0], e[1]) and not np.allclose(e[0], e[2])
    # origin maps to [sin 0 ..., cos 0 ...] = [0..,1..]
    assert np.allclose(e[0, :16], 0.0) and np.allclose(e[0, 16:32], 1.0)
    print("ok  spe: shape/bounds/determinism/origin")


def test_serialization():
    from gest.serialization import (
        crop_square, diagonal_serialize, neighbor_context, spatial_attention_mask)
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 100, size=(500, 2))

    idx = crop_square(coords, window=20.0, rng=rng, min_cells=10)
    assert idx is not None and idx.size >= 10
    sub = coords[idx]
    assert (np.ptp(sub, axis=0) <= 20.0 + 1e-6).all()        # within the window

    order = diagonal_serialize(sub, rng)
    assert np.array_equal(np.sort(order), np.arange(sub.shape[0]))  # a permutation

    # neighbor_context: nearest observed to a target at a known point
    obs = np.array([[0, 0], [10, 0], [0, 10], [100, 100]], dtype=float)
    tgt = np.array([[1, 1]], dtype=float)
    nb = neighbor_context(tgt, obs, k=2)
    assert nb.shape == (1, 2) and 0 in nb[0]                  # (0,0) is nearest

    print("ok  serialization: crop/permutation/neighbors")


def test_attention_mask():
    from gest.serialization import spatial_attention_mask
    L = 3
    M = spatial_attention_mask(L)
    assert M.shape == (6, 6)
    # NO all-False row (would -> NaN softmax)
    assert M.any(axis=1).all(), "an attention row is all-False (NaN softmax!)"
    # content rows causal: row j attends 0..j only
    for j in range(L):
        assert M[j, : j + 1].all() and not M[j, j + 1:].any()
    # target row i+L attends content 0..i-1 and target L..L+i-1, nothing later
    for i in range(1, L + 1):
        row = i - 1 + L
        assert M[row, 0:i].all()                              # content 1..i
        assert not M[row, i:L].any()                          # not future content
        assert M[row, L:L + i].all()                          # target prefix incl self
        assert not M[row, L + i:].any()                       # not future targets
    print("ok  attention mask matches Eq. 5 + no NaN rows")


def main():
    test_tokenizer()
    test_level_aggregation()
    test_spe()
    test_serialization()
    test_attention_mask()
    print("\nALL GeST NUMPY-CORE TESTS PASSED")


if __name__ == "__main__":
    main()
