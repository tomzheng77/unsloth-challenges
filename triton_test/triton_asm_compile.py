import triton
import triton.language as tl
import torch
from torch.library import custom_op

# Define Triton kernel with inline ASM
@triton.jit
def no_fma_kernel(a_ptr, b_ptr, c_ptr, out_ptr):
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    c = tl.load(c_ptr)
    result = a * b
    tl.store(out_ptr, result)
    result2 = tl.inline_asm_elementwise(
        asm="add.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[result, c],
        dtype=tl.float32,
        is_pure=True,
        pack=1
    )
    tl.store(out_ptr, result2)

# Define the custom op
@custom_op("mylib::triton_no_fma", mutates_args=())
def triton_no_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(a, device="cuda")
    grid = (1,)
    no_fma_kernel[grid](a, b, c, out)
    return out

# Register a fake implementation
@triton_no_fma.register_fake
def fake_triton_no_fma(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # Simulate the operation's semantics (shape and dtype) without CUDA execution
    return a * b + c  # Matches the kernel's logical output

# Register CUDA implementation (optional, but good practice)
# triton_no_fma.register_impl("cuda", triton_no_fma)

# Compiled function using the custom op
@torch.compile(fullgraph=True)
def torch_fn(a, b, c):
    return torch.ops.mylib.triton_no_fma(a, b, c)

# Test
a = torch.tensor([1.617633819580078125], dtype=torch.float32, device="cuda")
b = torch.tensor([0.31796872615814208984375], dtype=torch.float32, device="cuda")
c = torch.tensor([2.612834930419921875], dtype=torch.float32, device="cuda")

result = torch_fn(a, b, c)
torch.set_printoptions(precision=30)
print(result)  # Should match a * b + c
print(a * b + c)
