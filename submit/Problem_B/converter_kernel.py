import triton
import triton.language as tl
import torch


@triton.jit
def uint8_to_float16_block_kernel(
        input_ptr,  # Pointer to input uint8 tensor
        output_ptr,  # Pointer to output float16 tensor
        N: tl.constexpr,  # Total number of elements in the input tensor
        BLOCK_SIZE: tl.constexpr  # Number of elements each program processes (1024)
):
    # Get the program ID (unique for each thread block)
    pid = tl.program_id(axis=0)

    # Calculate the starting and ending indices for this program's block
    indices = pid * BLOCK_SIZE + (tl.arange(0, BLOCK_SIZE // 2) * 2)
    mask = indices < N
    values_uint8 = tl.load(input_ptr + indices)
    values_uint16 = values_uint8.to(tl.uint16)

    other_indices = pid * BLOCK_SIZE + (tl.arange(0, BLOCK_SIZE // 2) * 2) + 1
    other_uint8 = tl.load(input_ptr + other_indices)
    values_uint16 |= other_uint8.to(tl.uint16) << 8

    M = N // 2
    out_indices = pid * (BLOCK_SIZE // 2) + tl.arange(0, BLOCK_SIZE // 2)
    out_mask = out_indices < M
    tl.store(output_ptr + out_indices, values_uint16.to(tl.float16, bitcast=True))


# Example input tensor (uint8)
def convert_uint8_to_float16(input_tensor):
    # input_tensor = torch.arange(N, dtype=torch.uint8, device='cuda')
    N = input_tensor.numel()
    assert (N % 2 == 0)

    dims = list(input_tensor.shape)
    for i in range(len(dims)):
        if dims[i] != 1:
            dims[i] = dims[i] // 2
            break

    output_tensor = torch.empty(N // 2, dtype=torch.float16, device=input_tensor.device)

    # Define BLOCK_SIZE
    BLOCK_SIZE = 1024

    # Calculate the grid size (number of blocks needed)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    uint8_to_float16_block_kernel[grid](input_tensor, output_tensor, N, BLOCK_SIZE)
    return output_tensor.reshape(tuple(dims))


@triton.jit
def uint8_to_float32_block_kernel(
        input_ptr,  # Pointer to input uint8 tensor
        output_ptr,  # Pointer to output float32 tensor
        N: tl.constexpr,  # Total number of elements in the input tensor
        BLOCK_SIZE: tl.constexpr  # Number of elements each program processes (1024)
):
    # Get the program ID (unique for each thread block)
    pid = tl.program_id(axis=0)

    # Calculate the starting and ending indices for this program's block
    indices = pid * BLOCK_SIZE + (tl.arange(0, BLOCK_SIZE // 4) * 4)
    mask = indices < N
    values_uint8 = tl.load(input_ptr + indices)
    values_uint32 = values_uint8.to(tl.uint32)
    values_uint32 |= tl.load(input_ptr + indices + 1).to(tl.uint32) << 8
    values_uint32 |= tl.load(input_ptr + indices + 2).to(tl.uint32) << 16
    values_uint32 |= tl.load(input_ptr + indices + 3).to(tl.uint32) << 24

    M = N // 4
    out_indices = pid * (BLOCK_SIZE // 4) + tl.arange(0, BLOCK_SIZE // 4)
    out_mask = out_indices < M
    tl.store(output_ptr + out_indices, values_uint32.to(tl.float32, bitcast=True))


# Example input tensor (uint8)
def convert_uint8_to_float32(input_tensor):
    # input_tensor = torch.arange(N, dtype=torch.uint8, device='cuda')
    N = input_tensor.numel()
    assert (N % 4 == 0)

    dims = list(input_tensor.shape)
    result_numel = 1
    for i in range(len(dims)):
        if dims[i] != 1:
            dims[i] = dims[i] // 4
            break

    for d in dims: result_numel *= d
    output_tensor = torch.empty(N // 4, dtype=torch.float32, device=input_tensor.device)
    assert (output_tensor.numel() == result_numel)

    # Define BLOCK_SIZE
    BLOCK_SIZE = 1024

    # Calculate the grid size (number of blocks needed)
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the kernel
    uint8_to_float32_block_kernel[grid](input_tensor, output_tensor, N, BLOCK_SIZE)
    return output_tensor.reshape(tuple(dims))


if __name__ == '__main__':
    input_tensor = torch.arange(1024, dtype=torch.uint8, device='cuda').reshape(1, 1024)
    output_tensor = convert_uint8_to_float32(input_tensor)
    print(input_tensor)
    print(output_tensor)
