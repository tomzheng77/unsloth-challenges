import triton
import triton.language as tl
import torch


@triton.jit
def dequantize_4bit_kernel(
        # Pointers to matrices
        a_ptr, code_ptr, absmax_ptr, state2_code_ptr, state2_absmax_ptr, output_ptr,
        # Matrix dimensions
        M, N,
        # Parameters
        offset, blocksize, state2_blocksize,
        # Constants for Triton
        BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)

    # Calculate number of blocks
    num_blocks = N * 2 // blocksize

    # Check if this program ID is valid
    if pid >= num_blocks:
        return

    # Calculate the block start
    block_start = pid * blocksize

    # Load bytes for this block (each byte contains two 4-bit elements)
    byte_offset = block_start // 2
    n_bytes = blocksize // 2

    # Using block-level load (contiguous memory access pattern)
    bytes_block = tl.load(a_ptr + byte_offset + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < n_bytes)

    # Extract high and low 4-bits from each byte
    elm0 = (bytes_block >> 4) & 0xF  # High 4 bits
    elm1 = bytes_block & 0xF  # Low 4 bits

    # Load values from code table using the extracted indices
    val0 = tl.load(code_ptr + elm0)
    val1 = tl.load(code_ptr + elm1)

    # Get the absmax for this block
    absmax_idx = tl.load(absmax_ptr + pid)

    # Convert absmax_idx to integer type suitable for indexing
    absmax_idx = absmax_idx.to(tl.int32)

    # Load the corresponding state2 code value - explicitly dereference
    state2_code_val = tl.load(state2_code_ptr + absmax_idx)

    # Calculate which state2 absmax entry this block corresponds to
    state2_absmax_idx = pid // (state2_blocksize // blocksize)

    # Load the state2 absmax value - explicitly dereference
    state2_absmax_val = tl.load(state2_absmax_ptr + state2_absmax_idx)

    # Calculate final absmax (using scalar values)
    absmax_val = state2_code_val * state2_absmax_val + tl.load(offset)

    # Apply absmax scaling to values
    scaled_val0 = val0 * absmax_val
    scaled_val1 = val1 * absmax_val

    # Interleave values and convert to bfloat16
    # Handling first half (values from high 4 bits)
    for i in range(BLOCK_SIZE):
        if i < n_bytes:
            out_idx = block_start + i * 2
            if out_idx < N * 2:
                tl.store(output_ptr + out_idx, scaled_val0[i].to(tl.bfloat16))

    # Handling second half (values from low 4 bits)
    for i in range(BLOCK_SIZE):
        if i < n_bytes:
            out_idx = block_start + i * 2 + 1
            if out_idx < N * 2:
                tl.store(output_ptr + out_idx, scaled_val1[i].to(tl.bfloat16))


def dequantize_4bit_triton(A, quant_state):
    """
    Triton implementation of my_dequantize_4bit function

    Args:
        A: Input tensor in 4-bit quantized format
        quant_state: Quantization state with the following attributes:
            - code: [16] lookup table for 4-bit values
            - absmax: [num_blocks] quantized absmax values
            - state2.code: [256] lookup table for absmax
            - state2.absmax: [1024] scaling factors
            - state2.blocksize: 256
            - offset: scalar offset
            - blocksize: number of elements per block
            - shape: original shape of the tensor

    Returns:
        Dequantized tensor in bfloat16 format
    """
    # Ensure A is on GPU and has the right shape
    assert A.is_cuda and A.dim() == 2 and A.shape[0] == 1

    # Get dimensions
    M, N = 1, A.shape[1]
    blocksize = quant_state.blocksize

    # Calculate output shape
    output_shape = quant_state.shape
    output = torch.empty(output_shape, dtype=torch.bfloat16, device=A.device)

    # Determine optimal block size (power of 2, <= blocksize)
    block_size = min(128, blocksize // 2)  # Reduced block size for processing bytes

    # Number of blocks to process
    num_blocks = N * 2 // blocksize

    # Launch kernel
    grid = (num_blocks,)

    # Create flattened output for the kernel
    flat_output = output.t().reshape(-1)

    # Ensure all tensors are contiguous
    A_flat = A.reshape(-1).contiguous()

    # Ensure code tensors are contiguous and have the right shape
    code = quant_state.code.contiguous()
    absmax = quant_state.absmax.contiguous()
    state2_code = quant_state.state2.code.contiguous()
    state2_absmax = quant_state.state2.absmax.contiguous()

    # Print the shapes for debugging
    print(f"code shape: {code.shape}, absmax shape: {absmax.shape}")
    print(f"state2_code shape: {state2_code.shape}, state2_absmax shape: {state2_absmax.shape}")

    dequantize_4bit_kernel[grid](
        A_flat,
        code,
        absmax,
        state2_code,
        state2_absmax,
        flat_output,
        M, N,
        quant_state.offset,
        blocksize,
        quant_state.state2.blocksize,
        BLOCK_SIZE=block_size,
    )

    return output