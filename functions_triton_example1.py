import triton
import triton.language as tl
import torch

@triton.jit
def fused_dequantize_kernel(
    a_ptr,  # Input: packed 4-bit tensor (uint8)
    quant_absmax_ptr,  # Input: quant_state.absmax (uint8)
    state2_code_ptr,  # Input: quant_state.state2.code (float32)
    state2_absmax_ptr,  # Input: quant_state.state2.absmax (float32)
    code_ptr,  # Input: quant_state.code (float32)
    output_ptr,  # Output: dequantized result (bfloat16)
    offset,  # Input: quant_state.offset (float32)
    n_packed_elements,  # Number of uint8 elements in A
    blocksize,  # Elements per block (e.g., 256)
    BLOCK_SIZE: tl.constexpr  # Number of packed elements processed per thread block
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)
    out_block_idx = pid
    num_out_blocks = n_packed_elements * 2 // blocksize

    # Early return if block index is out of bounds
    if out_block_idx >= num_out_blocks:
        return

    # Compute the scaling factor for this block
    absmax_idx = out_block_idx * blocksize
    quant_absmax_val = tl.load(quant_absmax_ptr + absmax_idx).to(tl.int32)
    code_val = tl.load(state2_code_ptr + quant_absmax_val)
    state2_absmax_val = tl.load(state2_absmax_ptr + out_block_idx)
    scaling = code_val * state2_absmax_val + tl.load(offset)

    # Process packed elements for this block
    packed_per_block = blocksize // 2  # Number of uint8 elements per block
    packed_start = out_block_idx * packed_per_block
    packed_offsets = packed_start + tl.arange(0, BLOCK_SIZE)

    # Mask to prevent out-of-bounds access
    packed_mask = packed_offsets < n_packed_elements

    # Load packed uint8 values
    packed_vals = tl.load(a_ptr + packed_offsets, mask=packed_mask, other=0).to(tl.uint8)

    # Unpack 4-bit values
    val0 = (packed_vals >> 4).to(tl.int32)  # High 4 bits
    val1 = (packed_vals & 0b1111).to(tl.int32)  # Low 4 bits

    # Lookup dequantized values
    result0 = tl.load(code_ptr + val0, mask=packed_mask, other=0.0)
    result1 = tl.load(code_ptr + val1, mask=packed_mask, other=0.0)

    # Apply scaling
    result0 = result0 * scaling
    result1 = result1 * scaling

    # Compute output offsets (interleaved: val0, val1, val0, val1, ...)
    out_start = out_block_idx * blocksize
    out_offsets0 = out_start + 2 * tl.arange(0, BLOCK_SIZE)
    out_offsets1 = out_offsets0 + 1
    out_mask = out_offsets1 < (n_packed_elements * 2)

    # Store results as bfloat16
    tl.store(output_ptr + out_offsets0, result0.to(tl.bfloat16), mask=out_mask)
    tl.store(output_ptr + out_offsets1, result1.to(tl.bfloat16), mask=out_mask)


# Host function to launch the kernel
def triton_fused_dequantize(A, quant_state):
    """
    Fused dequantization of a 4-bit packed tensor using Triton.

    Args:
        A: torch.Tensor of shape [1, n_packed_elements], dtype uint8, containing packed 4-bit values
        quant_state: Object containing:
            - absmax: torch.Tensor of shape [total_elements], dtype uint8
            - state2.code: torch.Tensor, lookup table for absmax dequantization
            - state2.absmax: torch.Tensor of shape [num_blocks], per-block scaling
            - code: torch.Tensor, lookup table for 4-bit values
            - offset: float, scaling offset
            - blocksize: int, elements per block (e.g., 256)
            - shape: tuple, desired output shape (e.g., [M, N])

    Returns:
        torch.Tensor of shape quant_state.shape, dtype bfloat16
    """
    assert A.is_cuda, "Input tensor must be on CUDA"
    n_packed_elements = A.shape[1]
    output = torch.empty(quant_state.shape, dtype=torch.bfloat16, device="cuda").t().contiguous()

    blocksize = quant_state.blocksize
    num_out_blocks = (n_packed_elements * 2) // blocksize
    BLOCK_SIZE = blocksize // 2  # Number of packed elements per thread block

    grid = (num_out_blocks,)
    fused_dequantize_kernel[grid](
        A,
        quant_state.absmax,
        quant_state.state2.code,
        quant_state.state2.absmax,
        quant_state.code,
        output,
        quant_state.offset,
        n_packed_elements,
        blocksize,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output.t()  # Reshape to original dimensions
