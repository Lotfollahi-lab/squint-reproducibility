import torch
import math
from torch_geometric.nn.dense.linear import Linear
from torch.nn import Linear as PyTorchLinear

# Step 1: Set seed and save RNG state
device = torch.device('cpu')

torch.manual_seed(42)
cpu_rng = torch.get_rng_state()
cuda_rng = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

# Step 2: Create a Linear layer and reset
# layer = PyTorchLinear(in_features=128, out_features=64, bias=True)
layer = Linear(in_channels=128, out_channels=64, bias=True)

# Save weights and bias
w1 = layer.weight.detach().clone()
b1 = layer.bias.detach().clone()

# Step 3: Restore RNG
torch.set_rng_state(cpu_rng)
if torch.cuda.is_available():
    torch.cuda.set_rng_state(cuda_rng)

# Step 4: Reset again
layer.reset_parameters()

# Save second round of weights
w2 = layer.weight.detach().clone()
b2 = layer.bias.detach().clone()

# Step 5: Compare
print("Weight match:", torch.allclose(w1, w2, atol=1e-6))
print("Bias match  :", torch.allclose(b1, b2, atol=1e-6))

# Diagnostic output
print("Weight max diff:", (w1 - w2).abs().max().item())
print("Bias max diff:", (b1 - b2).abs().max().item())

with torch.no_grad():
    layer.weight.copy_(w1)
    layer.bias.copy_(b1)
    
print("Weight match after copy:", torch.allclose(layer.weight, w1, atol=1e-6))
print("Bias match after copy  :", torch.allclose(layer.bias, b1, atol=1e-6))
