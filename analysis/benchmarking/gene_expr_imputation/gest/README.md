# GeST (reimplementation) — spatial-imputation benchmark

Faithful reimplementation of **GeST** (Hao et al., *GeST: Towards Building A
Generative Pretrained Transformer for Learning Cellular Spatial Context*, MLCB
2025, PMLR v311 `hao25a`) for the gene-expression imputation benchmark.

**There is no official GeST code or pretrained weights** (verified: lead author
`whirlfirst`, lab org `XuegongLab`, GitHub repo search, HuggingFace, the PMLR
camera-ready, and the bioRxiv preprint — none release code/weights). So this is
a from-scratch reimplementation of the paper's *unseen cell generation* task =
spatial imputation: predict a held-out region's per-cell expression from the
surrounding cells. Used as the `GeST (imputed)` bar next to `SQUINT (imputed)`.

## Components (1:1 with the paper)

| file | paper ref | what |
|---|---|---|
| `tokenizer.py` | Eq. 7, 8 | meta-cell vocabulary (PCA→K-means→K meta cells; tokenize to nearest meta cell's mean expression) + 4-level hierarchy + projection of model output to meta-cell logits + weighted-aggregation / picking decode |
| `positional.py` | §3.3 | 2D sinusoidal spatial position encoding (SPE, ViT-style) |
| `serialization.py` | Eq. 5, 6 | square crop + diagonal-path distance-weighted ordering; the 2L×2L Spatial-Attention mask; inference neighbor gathering |
| `model.py` | §3.1, Eq. 9 | decoder-only transformer (RMSNorm blocks) + hierarchical cross-entropy loss (α=0.25) |
| `train.py` | §4.1 | per-dataset training loop + per-held-out-cell inference from observed neighbors |

## How it's used (vs the paper)

The paper *pretrains* GeST on a large corpus then evaluates zero-shot. With no
weights, we **train per-dataset** — exactly how the paper trains its own GP/MLP
baselines (Table 1): fit on the train (non-held-out) cells, predict the held-out
region. Each held-out cell is predicted from its k nearest **observed (train)**
neighbors (the model's trained conditional `P(g(x)|s(x),g(N(x)),s(N(x)))`); no
autoregressive error accumulation since the context is real observations.

Deliberate, documented simplifications:
- Crop window is in **median-NN-distance units** (not µm) for scale-invariance —
  mmb mixes MERFISH + STARmap with very different coordinate frames.
- Content-token rows in the attention mask use causal self-attention (the paper
  specifies only the target rows, Eq. 5; content rows otherwise softmax over
  all-`-inf` → NaN). Target readout is unchanged.
- Cell branch only (the figure's cell panel). Niche-level imputation needs
  neighborhood aggregation of the decoder output — a follow-up.

## Validation

- **numpy core** (`tests/test_numpy_core.py`): tokenizer blob recovery (meta
  purity + coarse-level recovery), level-aggregation probability conservation,
  SPE shape/bounds/determinism, crop/permutation/neighbor correctness, and the
  attention mask vs Eq. 5 (incl. no NaN rows). All pass locally.
- **torch model**: `py_compile` only locally (no torch in the dev env). The
  end-to-end correctness check runs on the farm: `run_gest.py` reports the
  **test gene-wise Spearman**, to be cross-checked against the paper's Table 1
  MERFISH ballpark (ρ ≈ 0.24–0.30). `--smoke` does a tiny fast plumbing run.

## Run

```bash
# plumbing
python run_gest.py --silver-dir <mmb silver> --smoke
# real (5 seeds)
python run_gest.py --silver-dir <mmb silver>
# -> <artifacts>/mmb0-1b_smb1-1b_1p/gest-imputed+region-holdout/<TS>/metrics/per_seed_pearson_reconstruction.csv
```
Then add the bar: `plot_pearson_benchmark.py --imputed gest-imputed+region-holdout "GeST (imputed)"`.
