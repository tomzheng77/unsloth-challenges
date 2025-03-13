import torch
import os

os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Define BF16 tensors on a T4 GPU
a = torch.tensor([1.2345], dtype=torch.bfloat16, device='cuda')
b = torch.tensor([2.3456], dtype=torch.bfloat16, device='cuda')

# Multiply
@torch.compile(fullgraph=True)
def mul_as_fp32(a, b):
    return a.to(torch.float32) * b.to(torch.float32)

# Print result
print(mul_as_fp32(a, b))
