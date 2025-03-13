import struct

import triton
import triton.language as tl


@triton.jit
def float32_to_bf16_kernel(
        input_ptr,  # Pointer to float32 input
        output_ptr,  # Pointer to bf16 output (stored as uint16)
        n_elements,  # Number of elements
        BLOCK_SIZE: tl.constexpr,
):
    # Compute the offset for this program instance
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load float32 input
    x = tl.load(input_ptr + offsets, mask=mask)

    # Reinterpret float32 as uint32 by casting through a memory-like operation
    # Since Triton lacks reinterpret, we use a cast to uint32 that preserves bits
    bits = tl.cast(x, tl.uint32, bitcast=True)  # Updated to use tl.bitcast

    # Shift right and truncate to 16 bits
    bf16_bits = (bits >> 16) & 0xffff

    # Store as uint16 (bf16 format)
    tl.store(output_ptr + offsets, bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True), mask=mask)


# Example usage
import torch


def float32_to_bf16_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.bfloat16, device=x.device)

    # Grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    float32_to_bf16_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


# Test
x = torch.tensor([1.234, 2.567], dtype=torch.float32, device="cuda")
bf16_output = float32_to_bf16_triton(x)
print(bf16_output)  # Outputs bf16 bits as uint16

def float32_to_bits(f: float) -> int:
    # Pack float32 into 4 bytes, then unpack as uint32
    packed = struct.pack('f', f)
    return struct.unpack('I', packed)[0]

print(float32_to_bits(1.234))
print(float32_to_bits(2.567))

f = 1.234
bits = float32_to_bits(f)
print(f"{f} -> {bits:#010x}")

print(list(float32_to_bf16_kernel.cache[0].values())[0].asm['ptx'])
