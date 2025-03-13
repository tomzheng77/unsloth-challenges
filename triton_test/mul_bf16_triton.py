import torch
import os
import triton
import triton.language as tl

os.environ['TORCH_COMPILE_DEBUG'] = '1'
os.environ['TRITON_DEBUG'] = '1'
os.environ["TORCHDYNAMO_VERBOSE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"


@triton.jit
def simple_mul_kernel(a_ptr, b_ptr, out_ptr):
    # Load single values from pointers
    a = tl.load(a_ptr).to(tl.float32)
    b = tl.load(b_ptr).to(tl.float32)

    # Store the result
    tl.store(out_ptr, a * b)


# Define BF16 tensors on a T4 GPU
a = torch.tensor([1.2345], dtype=torch.bfloat16, device='cuda')
b = torch.tensor([2.3456], dtype=torch.bfloat16, device='cuda')
out = torch.tensor([0], dtype=torch.float32, device='cuda')

grid = (1,)
simple_mul_kernel[grid](a, b, out)

print(out)
print(list(simple_mul_kernel.cache[0].values())[0].asm['ptx'])
