import sys
import torch.nn as nn
from pathlib import Path

# Add your source folder so Python can find your model definition
sys.path.append("/lustre/scratch126/cellgen/lotfollahi/am84/riley_jung/vqniche/src")

from vqniche.models.vqniche import VQNiche  # import model

# Load checkpoint
ckpt_path = "/lustre/scratch126/cellgen/lotfollahi/am84/riley_jung/log/xhs1000-39b_1p/standalone/VQNiche/batch=[2, 5, 10, 16]/spatial_n_neighs_8/wandb/offline-run-20251021_120930-7vgltmlc/files/checkpoints/epoch=0-train_pearson_1hop_nbr=0.74.ckpt"
model = VQNiche.load_from_checkpoint(ckpt_path, map_location="cpu")

# Print out all Linear layers and their dimensions
print("Listing all Linear layers and their in/out features:\n")
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        print(f"{name:60s} | in_features={module.in_features}, out_features={module.out_features}")
