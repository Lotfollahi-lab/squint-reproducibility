# 2D Imputation Task:

## Main Results Table (3 Test Regions)

- `./scripts/wrapper_sweep_train_model.sh mmb0-4b_1p 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh mmb0-4b_1p 3-test-patch-split vqniche_graphsage seed 0 6 gpu-lotfollahi-train`

- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-3 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-3 3-test-patch-split vqniche_graphsage seed 0 6 gpu-lotfollahi-train`

- `./scripts/wrapper_sweep_train_model.sh xhk1020-CV1-CV2-5b_1p 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhk1020-CV1-CV2-5b_1p 3-test-patch-split vqniche_graphsage seed 0 6 gpu-lotfollahi-train`


## Skin (Human) (3 Test Regions) -- Increasing Training Sections

- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-3 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-4 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-5 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-6 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-7 3-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`

## Skin (Human) (1 Test Region) -- Increasing Training Sections

- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-3 1-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-4 1-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-5 1-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-6 1-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`
- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-7 1-test-patch-split vqniche_graphsage seed 1 6 gpu-lotfollahi-train`

## Skin (Human) (1 Test Region) -- Increasing Imputation Patch Size

- `./scripts/wrapper_sweep_train_model.sh xhs1000-39b_1p-oriented-3 1-test-patch-split vqniche_graphsage patch_size 1 6 gpu-lotfollahi-train`
- 