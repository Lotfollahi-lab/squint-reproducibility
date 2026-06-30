"""
GeST meta-cell vocabulary + hierarchical labels (numpy / sklearn; no torch).

Paper refs (Hao et al., MLCB 2025):
  - Eq. 7  cell tokenization: PCA -> K-means into K "meta cells"; tokenize a
           cell to its nearest meta cell in PCA space, whose mean expression
           C_expr[i] is the cell's token.
  - Eq. 8  hierarchical labels at 4 levels l0..l3 with K, K1, K2, K3 categories
           (l0 = finest). Coarser levels = K-means on the meta-cell PCA centers.
  - Eq. 9  hierarchical cross-entropy: project model output yhat to meta-cell
           logits z = yhat @ C_expr^T, p(c)=softmax(z), aggregate to level-i
           probabilities p^(i)(k) = sum_c [l_i(c)==k] p(c), loss = sum_i a_i CE.

Decode (sec 3.4): "picking" = argmax meta cell's profile; "weighted
aggregation" = sum_c p(c) C_expr[c] (the better mode on MERFISH).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


def _to_dense(x) -> np.ndarray:
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


@dataclass
class MetaCellVocab:
    """Fitted meta-cell vocabulary.

    Attributes
    ----------
    C_expr : (K, T) mean expression profile per meta cell (the token g(x)).
    C_pca  : (K, P) mean PCA coordinate per meta cell (for nearest-meta lookup).
    level_labels : list of 4 arrays, each (K,) int — meta cell -> coarse label
                   at level i (level 0 is identity: arange(K)).
    level_sizes  : [K, K1, K2, K3].
    pca_mean_, pca_components_ : the PCA transform (fit on the training expr).
    log1p : whether expression was log1p'd before PCA (matches preprocessing).
    """

    C_expr: np.ndarray
    C_pca: np.ndarray
    level_labels: List[np.ndarray]
    level_sizes: List[int]
    pca_mean_: np.ndarray
    pca_components_: np.ndarray
    log1p: bool = True

    # ---- fit ----------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        n_meta: int = 500,
        level_sizes: Optional[List[int]] = None,
        n_pca: int = 50,
        log1p: bool = True,
        seed: int = 0,
    ) -> "MetaCellVocab":
        """Fit on a TRAIN expression matrix X (n_cells, T) of raw counts.

        n_meta = K (finest level). level_sizes overrides the coarser
        [K1, K2, K3] (defaults to K/4, K/16, K/64, floored at >=2).
        """
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        Xd = _to_dense(X)
        Xp = np.log1p(np.clip(Xd, 0, None)) if log1p else Xd

        n_pca = int(min(n_pca, Xp.shape[1], max(2, Xp.shape[0] - 1)))
        pca = PCA(n_components=n_pca, random_state=seed)
        Z = pca.fit_transform(Xp).astype(np.float32)        # (n, P)

        K = int(min(n_meta, Xp.shape[0]))
        km = KMeans(n_clusters=K, random_state=seed, n_init=10)
        assign = km.fit_predict(Z)                           # (n,)

        # meta-cell profiles: mean expression + mean PCA over assigned cells.
        T = Xd.shape[1]
        C_expr = np.zeros((K, T), dtype=np.float32)
        C_pca = np.zeros((K, n_pca), dtype=np.float32)
        for k in range(K):
            m = assign == k
            if m.any():
                C_expr[k] = Xd[m].mean(axis=0)
                C_pca[k] = Z[m].mean(axis=0)
            else:                                            # empty cluster -> centroid
                C_pca[k] = km.cluster_centers_[k]

        # hierarchical coarse labels: K-means on the meta-cell PCA centers.
        if level_sizes is None:
            level_sizes = [max(2, K // 4), max(2, K // 16), max(2, K // 64)]
        sizes = [K] + [int(min(s, K)) for s in level_sizes]
        labels = [np.arange(K, dtype=np.int64)]              # level 0 = identity
        for s in sizes[1:]:
            if s >= K:
                labels.append(np.arange(K, dtype=np.int64))
                continue
            km_c = KMeans(n_clusters=s, random_state=seed, n_init=10)
            labels.append(km_c.fit_predict(C_pca).astype(np.int64))

        return cls(
            C_expr=C_expr, C_pca=C_pca, level_labels=labels, level_sizes=sizes,
            pca_mean_=pca.mean_.astype(np.float32),
            pca_components_=pca.components_.astype(np.float32),
            log1p=log1p,
        )

    # ---- transforms ---------------------------------------------------------
    @property
    def n_meta(self) -> int:
        return self.C_expr.shape[0]

    @property
    def n_genes(self) -> int:
        return self.C_expr.shape[1]

    def to_pca(self, X: np.ndarray) -> np.ndarray:
        """Project raw-count expression X (n, T) into the fitted PCA space."""
        Xd = _to_dense(X)
        Xp = np.log1p(np.clip(Xd, 0, None)) if self.log1p else Xd
        return ((Xp - self.pca_mean_[None, :]) @ self.pca_components_.T).astype(np.float32)

    def tokenize(self, X: np.ndarray) -> np.ndarray:
        """Eq. 7: nearest meta-cell index per cell (argmin PCA distance)."""
        Z = self.to_pca(X)                                   # (n, P)
        # squared euclidean to each meta center, vectorised
        d2 = (
            (Z ** 2).sum(1, keepdims=True)
            - 2.0 * Z @ self.C_pca.T
            + (self.C_pca ** 2).sum(1)[None, :]
        )
        return np.argmin(d2, axis=1).astype(np.int64)        # (n,)

    def token_expr(self, X: np.ndarray) -> np.ndarray:
        """g(x): the nearest meta cell's expression profile per cell (n, T)."""
        return self.C_expr[self.tokenize(X)]

    def level_label(self, meta_idx: np.ndarray, level: int) -> np.ndarray:
        """Coarse label at `level` for given fine meta-cell indices."""
        return self.level_labels[level][meta_idx]

    def labels_all_levels(self, X: np.ndarray) -> List[np.ndarray]:
        """Per-cell ground-truth label at each of the 4 levels (for the loss)."""
        fine = self.tokenize(X)
        return [self.level_labels[i][fine] for i in range(len(self.level_sizes))]
