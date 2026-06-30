"""
Numpy-only tests for the GeST-arch-on-SQUINT-codes wiring (no torch).

Validates the torch-free parts of the new ablation arm:
  * code-stack reading from a fake predicted_adata (obsm/uns conventions),
  * target-order assembly + per-target sizes/names,
  * the "unknown/mask" row indexing for held-out target slots (the exact index
    arithmetic the torch _embed_codes performs),
  * CodeStackSpec cell-level selection,
  * serialization reuse on a code-stack crop (diagonal order + suffix mask).

Run:  python -m gest.tests.test_squint_codes_np
The torch model (GeSTSquintCodes) is py_compile-checked + farm-smoke-tested only.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# A minimal stand-in for an AnnData carrying the frozen SQUINT codes, so we can
# test read_squint_codes without anndata installed.
# --------------------------------------------------------------------------- #
class _FakeAData:
    def __init__(self, n, obsm, uns):
        self.n_obs = n
        self.obsm = obsm
        self.uns = uns


def test_read_codes_obsm_aliases():
    from gest.squint_codes_np import (
        read_squint_codes, build_target_codes_array, per_target_sizes, target_names,
    )
    n = 40
    rng = np.random.default_rng(0)
    cell = np.stack([rng.integers(0, 30, n), rng.integers(0, 90, n)], axis=1)
    niche = np.stack([rng.integers(0, 30, n), rng.integers(0, 90, n)], axis=1)
    ad = _FakeAData(
        n,
        obsm={"cell_code_indices": cell, "neighborhood_code_indices": niche},
        uns={"codebook_sizes_cell": np.array([30, 90]),
             "codebook_sizes_niche": np.array([30, 90])},
    )
    cc, cn, sc, sn = read_squint_codes(ad)
    assert cc.shape == (n, 2) and cn.shape == (n, 2)
    assert sc == [30, 90] and sn == [30, 90]

    stack = build_target_codes_array(cc, cn)
    assert stack.shape == (n, 4)
    # column order is [cell L0, cell L1, niche L0, niche L1]
    assert np.array_equal(stack[:, :2], cc) and np.array_equal(stack[:, 2:], cn)
    assert per_target_sizes(sc, sn) == [30, 90, 30, 90]
    assert target_names(sc, sn) == ["cell.L0", "cell.L1", "niche.L0", "niche.L1"]

    # niche fallback alias + size inference from data when uns missing
    ad2 = _FakeAData(
        n,
        obsm={"cell_code_indices": cell, "niche_code_indices": niche},
        uns={},
    )
    cc2, cn2, sc2, sn2 = read_squint_codes(ad2)
    assert np.array_equal(cn2, niche)
    # inferred sizes are max+1 (>= the true max index)
    assert all(sc2[q] >= int(cell[:, q].max()) + 1 for q in range(2))
    print("ok  read_squint_codes: obsm aliases + size resolution + target order")


def test_mask_unknown_row_indexing():
    """Held-out target slots must index the per-target 'unknown' row == K_t.
    This is the exact index the torch model's nn.Embedding(K_t+1, d) reserves."""
    from gest.squint_codes_np import mask_target_codes, clamp_observed_codes
    sizes = [30, 90, 30, 90]
    n = 7
    rng = np.random.default_rng(1)
    obs = np.stack([rng.integers(0, k, n) for k in sizes], axis=1)

    masked = mask_target_codes(obs, sizes)
    assert masked.shape == obs.shape
    for t, k in enumerate(sizes):
        assert np.all(masked[:, t] == k), f"target {t} mask row should be {k}"
        # the unknown row is a VALID index into Embedding(K+1): 0..K
        assert masked[:, t].max() <= k

    # observed codes clamp into [0, K-1] (never collide with the mask row K)
    dirty = obs.copy()
    dirty[0, 0] = 999          # stray out-of-range
    dirty[1, 1] = -5
    clamped = clamp_observed_codes(dirty, sizes)
    for t, k in enumerate(sizes):
        assert clamped[:, t].min() >= 0 and clamped[:, t].max() <= k - 1
    print("ok  mask/clamp: unknown row == K_t, observed clamped to [0,K-1]")


def test_codestackspec_cell_levels():
    from gest.squint_codes_np import CodeStackSpec
    spec = CodeStackSpec.from_branch_sizes([30, 90], [30, 90])
    assert spec.n_targets == 4
    assert spec.names == ["cell.L0", "cell.L1", "niche.L0", "niche.L1"]
    assert spec.sizes == [30, 90, 30, 90]
    # cell_levels selects the two cell-branch columns (for the decoder)
    assert spec.cell_levels == [0, 1]

    # asymmetric stack depths still resolve correctly
    spec2 = CodeStackSpec.from_branch_sizes([16], [8, 24, 48])
    assert spec2.names == ["cell.L0", "niche.L0", "niche.L1", "niche.L2"]
    assert spec2.cell_levels == [0]
    print("ok  CodeStackSpec: target order + cell-level selection")


def test_serialization_reuse_on_codes():
    """The arm reuses GeST diagonal serialization + the suffix mask unchanged.
    Confirm a code-stack crop is ordered + that the mask wiring lines up with the
    content/target split the model uses (first L = content, last L = targets)."""
    from gest.serialization import diagonal_serialize, spatial_attention_mask
    rng = np.random.default_rng(2)
    n = 20
    coords = rng.random((n, 2)) * 100.0
    codes = np.stack([rng.integers(0, 30, n), rng.integers(0, 90, n)], axis=1)

    order = diagonal_serialize(coords, rng)
    assert order.shape == (n,) and sorted(order.tolist()) == list(range(n))
    # reordering the code stack by `order` is a pure gather (no value change)
    reordered = codes[order]
    assert reordered.shape == codes.shape
    assert set(map(tuple, reordered.tolist())) == set(map(tuple, codes.tolist()))

    seq_n = 8
    L = seq_n - 1
    M = spatial_attention_mask(L)
    assert M.shape == (2 * L, 2 * L)
    # content slots are the first L cells of the order; targets are cells 1..seq_n-1.
    content = order[:L]
    targets = order[1:seq_n]
    assert content.shape == (L,) and targets.shape == (L,)
    # every attention row has at least one True (no NaN softmax row)
    assert M.any(axis=1).all()
    print("ok  serialization reuse: diagonal order is a permutation + suffix mask intact")


def main():
    test_read_codes_obsm_aliases()
    test_mask_unknown_row_indexing()
    test_codestackspec_cell_levels()
    test_serialization_reuse_on_codes()
    print("\nALL GeST-SQUINT-CODES NUMPY TESTS PASSED")


if __name__ == "__main__":
    main()
