# Legacy configurations

The YAML files in this folder belong to the earlier single-codebook pipeline
(`analysis/train_model.py`, `scripts/sweep_train_model.sh`) and cover datasets that
the MLCB 2026 paper does not use. They are not the paper's configurations.

The SQUINT variants reported in the paper are defined in the `VARIANTS` registry of
squint's `examples/run_squint.py`, and every run writes its full configuration to
`<run>/user_specified_config.yaml`. See the top-level README.
