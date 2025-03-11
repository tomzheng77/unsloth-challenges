import torch
import triton
import triton.language as tl


# Define the Triton kernel for element-wise addition
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID (which block this thread is handling)
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE

    # Create an offset range for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Mask to prevent out-of-bounds memory access
    mask = offsets < n_elements

    # Load data from input arrays x and y
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Perform the addition
    output = x + y

    # Store the result in the output array
    tl.store(output_ptr + offsets, output, mask=mask)


# Function to run the kernel
def add_arrays(x: torch.Tensor, y: torch.Tensor):
    # Ensure inputs are on GPU and contiguous
    x = x.cuda().contiguous()
    y = y.cuda().contiguous()

    # Allocate output tensor
    output = torch.empty_like(x)

    # Number of elements in the arrays
    n_elements = x.numel()

    # Define block size (tunable parameter)
    BLOCK_SIZE = 1024

    # Calculate the grid size (number of blocks)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    # Launch the kernel
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE)

    return output


# Test the addition
if __name__ == "__main__":
    # Create sample input tensors
    size = 4096
    x = torch.randn(size, dtype=torch.float32).to('cuda')
    y = torch.randn(size, dtype=torch.float32).to('cuda')

    # Run the Triton addition
    result = add_arrays(x, y)

    # Verify with PyTorch (CPU/GPU equivalent)
    expected = x + y

    # Check correctness
    print("Result:", result[:10])  # Print first 10 elements
    print("Expected:", expected[:10])
    print("Max difference:", torch.max(torch.abs(result - expected)))
