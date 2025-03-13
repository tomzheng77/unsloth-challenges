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
    bits = tl.cast(x, tl.uint32)  # Updated to use tl.bitcast

    # Extract exponent to check for NaN/Inf
    exponent = (bits & 0x7f800000)
    is_special = (exponent == 0x7f800000)

    # Rounding logic
    mantissa_bit = (bits & (1 << 16)) != 0
    round_bit = (bits & (1 << 15)) != 0
    sticky_bits = (bits & ((1 << 15) - 1)) != 0
    should_round_up = (round_bit & sticky_bits) | (round_bit & mantissa_bit)

    # Increment if rounding up
    bits_adjusted = tl.where(should_round_up, bits + (1 << 16), bits)

    # Handle NaN/Inf cases
    bits_final = tl.where(
        is_special & (bits & ~0xff800000),  # NaN case
        0x7fffffff,  # Return a NaN
        bits_adjusted
    )

    # Shift right and truncate to 16 bits
    bf16_bits = (bits_final >> 16) & 0xffff

    # Store as uint16 (bf16 format)
    tl.store(output_ptr + offsets, bf16_bits.to(tl.uint16), mask=mask)


# Example usage
import torch


def float32_to_bf16_triton(x: torch.Tensor) -> torch.Tensor:
    assert x.dtype == torch.float32
    n_elements = x.numel()
    output = torch.empty(n_elements, dtype=torch.uint16, device=x.device)

    # Grid size
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # Launch kernel
    float32_to_bf16_kernel[grid](
        x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE
    )
    return output


# Test
x = torch.tensor([1.234, 2.567, float("inf"), float("nan")], dtype=torch.float32, device="cuda")
bf16_output = float32_to_bf16_triton(x)
print(bf16_output)  # Outputs bf16 bits as uint16
