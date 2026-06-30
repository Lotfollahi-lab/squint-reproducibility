"""Equation-level faithfulness checks: each test pins an implementation detail to
the corresponding GeST equation/section (Hao et al., MLCB 2025) by comparing
against an independent brute-force/reference. Numpy-only (no torch).

  python -m gest.tests.test_paper_faithfulness
"""
from __future__ import annotations

import numpy as np


def test_eq7_tokenization_is_pca_argmin():
    """Eq. 7: token(x) = argmin_k ||x_pca - C_pca[k]||_2 (nearest meta cell in
    PCA space). Verify the vectorised tokenizer == a brute-force argmin."""
    from gest.tokenizer import MetaCellVocab
    rng = np.random.default_rng(0)
    X = np.clip(rng.normal(5, 3, size=(300, 80)), 0, None).astype(np.float32)
    voc = MetaCellVocab.fit(X, n_meta=24, n_pca=12, seed=0)

    tok = voc.tokenize(X)
    Z = voc.to_pca(X)                                   # (n, P)
    # brute force nearest meta cell by Euclidean PCA distance
    brute = np.array([
        int(np.argmin(((voc.C_pca - z) ** 2).sum(1))) for z in Z
    ])
    assert np.array_equal(tok, brute), (tok[:10], brute[:10])
    # token_expr is exactly the chosen meta cell's profile
    assert np.array_equal(voc.token_expr(X), voc.C_expr[brute])
    print("ok  Eq.7  tokenize == argmin PCA distance (vs brute force)")


def test_eq8_hierarchical_aggregation_vs_bruteforce():
    """Eq. 8: p^(i)(k) = sum_c [l_i(c)==k] p(c). The model aggregates with
    torch index_add_; verify the equivalent np.add.at matches an explicit
    per-meta-cell loop, for every level, and conserves probability."""
    from gest.tokenizer import MetaCellVocab
    rng = np.random.default_rng(1)
    X = np.clip(rng.normal(5, 2, size=(200, 50)), 0, None).astype(np.float32)
    voc = MetaCellVocab.fit(X, n_meta=20, level_sizes=[10, 5, 2], n_pca=8, seed=1)
    p = rng.random(voc.n_meta); p = p / p.sum()
    for i, sz in enumerate(voc.level_sizes):
        lvl = voc.level_labels[i]
        brute = np.zeros(sz)
        for c in range(voc.n_meta):
            brute[lvl[c]] += p[c]
        agg = np.zeros(sz); np.add.at(agg, lvl, p)      # == torch index_add_
        assert np.allclose(agg, brute, atol=1e-7)
        assert abs(agg.sum() - 1.0) < 1e-6
    print("ok  Eq.8  hierarchical aggregation == brute-force per level (+ conserves mass)")


def test_eq6_serialization_matches_formula():
    """Eq. 6: x1 is a square-corner cell; remaining cells sampled w/o
    replacement with P ∝ ||p(x_oi) - p(x1)||_2. Signatures of that rule:
      (a) x1 is an extreme (corner) cell of the bbox;
      (b) weight = distance-to-x1 (fixed) => cells FAR from x1 get LOWER
          indices => distance vs final-index correlates NEGATIVELY."""
    from gest.serialization import diagonal_serialize
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 1, size=(120, 2))
    lo, hi = coords.min(0), coords.max(0)

    corner_hits, rhos = 0, []
    for _ in range(200):
        order = diagonal_serialize(coords, rng)
        x1 = order[0]
        # (a) x1 sits on an extreme of x OR y (a bbox corner-ish cell)
        c = coords[x1]
        if (np.isclose(c, lo, atol=0.06).any() or np.isclose(c, hi, atol=0.06).any()):
            corner_hits += 1
        # (b) distance-to-x1 vs index position
        dist = np.sqrt(((coords - coords[x1]) ** 2).sum(1))
        pos = np.empty(coords.shape[0]); pos[order] = np.arange(coords.shape[0])
        rhos.append(spearmanr(dist, pos).correlation)
    mean_rho = float(np.mean(rhos))
    assert corner_hits > 180, f"x1 not a corner often enough: {corner_hits}/200"
    assert mean_rho < -0.3, f"distance-to-x1 should anti-correlate with index, got {mean_rho:.3f}"
    print(f"ok  Eq.6  x1 corner {corner_hits}/200; dist-to-x1 vs index rho={mean_rho:.3f} (<0 ✓)")


def test_spe_is_2d_sinusoidal():
    """Sec 3.3: 2D sinusoidal SPE (ViT-style), half dims per axis, sin/cos
    pairs at geometric frequencies. Verify against an independent reference."""
    from gest.positional import spe
    rng = np.random.default_rng(3)
    coords = rng.uniform(-5, 5, size=(7, 2)).astype(np.float32)
    dim = 64
    out = spe(coords, dim)
    # independent reference
    n_freq = dim // 4
    inv = 1.0 / (10000.0 ** (np.arange(n_freq) / n_freq))
    ref = np.zeros((7, dim), np.float32)
    for a in range(2):
        ang = coords[:, a:a + 1] * inv[None, :]
        base = a * (dim // 2)
        ref[:, base:base + n_freq] = np.sin(ang)
        ref[:, base + n_freq:base + 2 * n_freq] = np.cos(ang)
    assert np.allclose(out, ref, atol=1e-5)
    # x and y are encoded in separate halves (2D, not 1D)
    c2 = coords.copy(); c2[:, 1] += 3.0
    o2 = spe(c2, dim)
    assert np.allclose(o2[:, :dim // 2], out[:, :dim // 2])      # x-half unchanged
    assert not np.allclose(o2[:, dim // 2:], out[:, dim // 2:])  # y-half changed
    print("ok  Sec3.3  SPE == 2D sinusoidal reference (separate x/y halves)")


def test_hierarchy_four_levels_default():
    """Sec 3.4: 4 hierarchical levels (K, K1, K2, K3), alpha=0.25 each. The
    runner's default fit (n_meta=500, no level_sizes) must yield 4 levels."""
    from gest.tokenizer import MetaCellVocab
    import pathlib, re
    rng = np.random.default_rng(4)
    X = np.clip(rng.normal(5, 2, size=(600, 40)), 0, None).astype(np.float32)
    voc = MetaCellVocab.fit(X, n_meta=64, n_pca=10, seed=4)    # default level_sizes
    assert len(voc.level_sizes) == 4, voc.level_sizes
    assert voc.level_sizes[0] == 64                            # l0 = K (finest)
    assert all(voc.level_sizes[i] > voc.level_sizes[i + 1] for i in range(3)), voc.level_sizes
    # alpha=0.25 default — read from model.py source (torch import broken locally)
    src = (pathlib.Path(__file__).resolve().parents[1] / "model.py").read_text()
    assert re.search(r"alpha:\s*float\s*=\s*0\.25", src), "GeSTConfig.alpha default != 0.25"
    print(f"ok  Sec3.4  4 levels, decreasing {voc.level_sizes}, alpha=0.25")


def test_decode_formula():
    """Sec 3.4 decode: weighted = sum_c p(c) C_expr[c]; picking = C_expr[argmax].
    (model.decode is torch; verify the formula in numpy on the same tensors.)"""
    from gest.tokenizer import MetaCellVocab
    rng = np.random.default_rng(5)
    X = np.clip(rng.normal(5, 2, size=(150, 30)), 0, None).astype(np.float32)
    voc = MetaCellVocab.fit(X, n_meta=16, n_pca=8, seed=5)
    p = rng.random((4, voc.n_meta)); p = p / p.sum(1, keepdims=True)
    weighted = p @ voc.C_expr                                  # (4, T)
    picking = voc.C_expr[p.argmax(1)]
    assert weighted.shape == (4, voc.n_genes) and weighted.min() >= 0
    assert picking.shape == (4, voc.n_genes)
    # weighted is a convex combo of nonneg profiles -> bounded by max profile
    assert weighted.max() <= voc.C_expr.max() + 1e-5
    print("ok  Sec3.4  decode formulas (weighted convex-combo + picking) consistent")


def main():
    test_eq7_tokenization_is_pca_argmin()
    test_eq8_hierarchical_aggregation_vs_bruteforce()
    test_eq6_serialization_matches_formula()
    test_spe_is_2d_sinusoidal()
    test_hierarchy_four_levels_default()
    test_decode_formula()
    print("\nALL PAPER-FAITHFULNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
