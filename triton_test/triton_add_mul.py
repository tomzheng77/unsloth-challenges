import triton
import triton.language as tl


@triton.jit
def simple_fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr, out2_ptr):
    # Load single values from pointers
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    c = tl.load(c_ptr)

    # Compute: result = a * b + c
    result = a * b

    # result2 = tl.add(result, c)
    # Inline PTX to force add.f32
    result2 = tl.inline_asm_elementwise(
        asm="add.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[result, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

    # Store the result
    tl.store(out_ptr, result)
    tl.store(out2_ptr, result2)


# Example usage with PyTorch
import torch

# Input scalars as 1-element tensors (pointers to single numbers)
a = torch.tensor(1.617633819580078125000000000000, dtype=torch.float32, device="cuda")
b = torch.tensor(0.317968726158142089843750000000, dtype=torch.float32, device="cuda")
c = torch.tensor(2.612834930419921875000000000000, dtype=torch.float32, device="cuda")
out = torch.tensor(0, dtype=torch.float32, device="cuda")
out2 = torch.tensor(0, dtype=torch.float32, device="cuda")

# Launch kernel with a single-thread grid
grid = (1,)  # 1 block, 1 thread
simple_fma_kernel[grid](a, b, c, out, out2)

@torch.compile(fullgraph=True)
def ze_torch(a, b, c):
    return a * b + c

torch.set_printoptions(precision=30)
print(out)  # Expected: [7.0] (2.0 * 3.0 + 1.0 = 7.0)
print(a * b)
print(out2)
print(a * b + c)
print(ze_torch(a, b, c))

print(dir(simple_fma_kernel.cache))
print(list(simple_fma_kernel.cache[0].values())[0].asm['ptx'])
