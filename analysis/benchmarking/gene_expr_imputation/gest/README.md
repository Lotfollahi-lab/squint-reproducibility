# GeST reimplementation

Reimplementation of **GeST** (Hao et al., *GeST: Towards Building A Generative
Pretrained Transformer for Learning Cellular Spatial Context*, MLCB 2025, PMLR
v311 `hao25a`), the imputation baseline of Table 2 of the SQUINT paper.

No code or pretrained weights of GeST have been released (we checked GitHub,
Hugging Face, the PMLR paper and the bioRxiv preprint). This folder is therefore
a from-scratch reimplementation of the paper's *unseen cell generation* task,
i.e. spatial imputation: predict the expression of the cells in a held-out
region from the surrounding observed cells. How it is run for the paper, and
how its output is scored, is described in Section 3.2 of the
[repository README](../../../../README.md), which takes precedence over this
file.

## Components

| File | Paper reference | Content |
|---|---|---|
| `tokenizer.py` | Eq. 7, 8 | Meta-cell vocabulary (PCA, then k-means to K meta cells; a cell is tokenised to its nearest meta cell's mean expression), 4-level hierarchy, projection of the model output to meta-cell logits, weighted-aggregation and picking decodes |
| `positional.py` | §3.3 | 2-D sinusoidal spatial position encoding (SPE, ViT-style) |
| `serialization.py` | Eq. 5, 6 | Square crop and diagonal-path, distance-weighted ordering; the 2L×2L spatial-attention mask; neighbour gathering at inference |
| `model.py` | §3.1, Eq. 9 | Decoder-only transformer (RMSNorm blocks) and hierarchical cross-entropy loss (α = 0.25) |
| `train.py` | §4.1 | Per-dataset training loop and per-held-out-cell inference from observed neighbours |
| `squint_codes.py`, `squint_codes_np.py`, `squint_codes_train.py` | | The GeST architecture trained on SQUINT's frozen codes (`gestarch` in `submit_imputation_recon.sh`, `run_gest_arch_squint.py`); an ablation that is not reported in the paper |

## How it is used

The paper pretrains GeST on a large corpus and evaluates it zero-shot. Without
released weights, we **train it per dataset**, as the paper trains its own GP
and MLP baselines: the meta-cell vocabulary and the model are fitted on the
observed (`data_split == "train"`) cells only, and each held-out cell is
predicted from its k nearest observed cells (the model's trained conditional
`P(g(x) | s(x), g(N(x)), s(N(x)))`). There is no autoregressive error
accumulation, since the context consists of real observations.

Deliberate simplifications:

- The crop window is in units of the median nearest-neighbour distance rather
  than µm, so it does not depend on the coordinate scale (the Mouse Brain data
  combine MERFISH and STARmap sections with very different coordinate frames).
- Content-token rows of the attention mask use causal self-attention. The paper
  specifies only the target rows (Eq. 5), and content rows would otherwise
  apply a softmax over a row of `-inf` and return NaN. The target read-out is
  unchanged.
- GeST predicts per-cell profiles. As for the other cell-level baselines, its
  neighbourhood-level prediction in Table 2 is the mean of the cell-level
  predictions over the per-section 16-NN spatial graph (`--nbr-neighs`;
  `add_neighborhood_layers` in `../_holdout_utils.py`).

Stabilisations relative to the literal paper (each documented in the code):

- `log1p` of the neighbour meta-profiles before the gene-embedding linear layer
  (raw counts have a large dynamic range and destabilise it). Token identities
  are unchanged.
- A `1/sqrt(T)` temperature on the meta-cell logits `z = yhat · C_expr^T`
  (`C_expr` holds raw counts, so unscaled logits saturate the softmax at
  initialisation). The units of the decoded profile are unchanged.
- At most `2 * seq_n` cells per crop enter the diagonal serialisation. The
  serialiser is an O(n²) Python loop, and a dense window can hold thousands of
  cells; only the first `seq_n` are kept anyway, so the connected-patch
  semantics are unchanged and the cost is bounded.

**Library size.** The default (`--decode weighted`) prediction is a
softmax-weighted mean of meta-cell profiles, in counts, computed from observed
cells. It is not rescaled to the held-out cell's library size, so the held-out
cell's own counts are never used. This is the leak-free setting under which
all imputation results of Table 2 are scored (SQUINT is decoded with the mean
library size of the 16 nearest observed cells, `--read-depth-mode neighbor`).
All methods are scored by the same code (`build_pearson_dataframe` in
`../_holdout_utils.py`): Pearson and Spearman correlation on log1p counts, RMSE
on counts and expressed-gene detection.

## Tests

The numpy parts (tokeniser, hierarchy aggregation, SPE, crop, ordering,
neighbour gathering, the attention mask against Eq. 5, and the equation-level
checks of `test_paper_faithfulness.py`) are tested without torch:

```bash
cd analysis/benchmarking/gene_expr_imputation
python -m pytest gest/tests
```

The torch model is exercised end to end by `run_gest.py --smoke`.

## Run

For the paper, GeST is launched with the other Table 2 baselines
(`METHODS=gest` in `submit_imputation_recon.sh`; see Section 3.2 of the
repository README). Directly, in the `squint` environment:

```bash
cd analysis/benchmarking/gene_expr_imputation
# plumbing check (tiny model, one seed)
python run_gest.py --silver-dir <Mouse Brain silver folder> --artifacts-root $ART --smoke
# seeds 0-4, as reported
python run_gest.py --silver-dir <Mouse Brain silver folder> --artifacts-root $ART
# -> $ART/mmb0-1b_smb1-1b_1p/gest-imputed+region-holdout/<timestamp>/metrics/per_seed_pearson_reconstruction.csv
```

Each run applies the shared region holdout (`DEFAULT_HOLDOUT_REGIONS` in
`../_holdout_utils.py`; check that it prints `test=6612`) and writes
`predicted_adata_seed<N>.h5ad` for each seed into the run folder. The Table 2 panels read this
folder by default (`plot_imputation_benchmark.py`, `--gest-imputed-path`).
