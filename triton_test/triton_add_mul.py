import triton
import triton.language as tl


@triton.jit
def simple_fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr):
    # Load single values from pointers
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    c = tl.load(c_ptr)

    # Compute: result = a * b + c
    result = a * b + c

    # Store the result
    tl.store(out_ptr, result)


# Example usage with PyTorch
import torch

# Input scalars as 1-element tensors (pointers to single numbers)
a = torch.tensor(1.617633819580078125000000000000, dtype=torch.float32, device="cuda")
b = torch.tensor(0.317968726158142089843750000000, dtype=torch.float32, device="cuda")
c = torch.tensor(2.612834930419921875000000000000, dtype=torch.float32, device="cuda")
out = torch.tensor(0, dtype=torch.float32, device="cuda")

# Launch kernel with a single-thread grid
grid = (1,)  # 1 block, 1 thread
simple_fma_kernel[grid](a, b, c, out)

torch.set_printoptions(precision=30)
print(out)  # Expected: [7.0] (2.0 * 3.0 + 1.0 = 7.0)
print(a * b + c)
