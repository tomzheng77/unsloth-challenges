import inspect

import torch
from bitsandbytes import functional

ENABLE_ASSERTIONS = False

# NOTE: torch.compile works in run, but not in debug mode, because debug calls sys._getiframe,
# which causes a graph break
@torch.compile(fullgraph=True)
def derive_absmax(A, quant_state):
    # maybe is better to just implement this
    # first lets dequantize the absmax. we need a total of 262144 values
    # we have as input:
    # quant_state.absmax: [262144] (uint8)
    # quant_state.offset: [] (float32)
    # quant_state.state2.absmax: [1024] (float32)
    # quant_state.state2.blocksize: 256
    # quant_state.state2.code: [256] (float32) (??)
    # see call to dequantize_blockwise(quant_state.absmax, quant_state.state2)
    # where do I find the CUDA code again
    # - looking at kDequantizeBlockwise in CUDA I don't think the "code" is used
    # could be really simple:
    # multiply quant_state.absmax by quant_state.state2.code (actually not mul but access code with absmax),
    values = torch.index_select(quant_state.state2.code, dim=0, index=quant_state.absmax.to(torch.long))

    # Reshape large_tensor to [1024, 256]
    # Reshape small_tensor to [1024, 1] for broadcasting
    # Multiply with broadcast
    small_tensor_length = quant_state.state2.absmax.shape[0]
    large_tensor_reshaped = values.view(small_tensor_length, quant_state.state2.blocksize)
    small_tensor_reshaped = quant_state.state2.absmax.view(small_tensor_length, 1)
    result = large_tensor_reshaped * small_tensor_reshaped
    result = result.view(-1)
    return result + quant_state.offset

@torch.compile(fullgraph=True)
def my_dequantize_4bit(A, quant_state):
    absmax = derive_absmax(A, quant_state)
    if ENABLE_ASSERTIONS:
        assert(absmax.dtype == torch.float32)
        assert(A.shape[0] == 1)
        assert(len(A.shape) == 2)
    elm0 = (A >> 4).to(torch.long).reshape(-1)
    elm1 = (A & 0b1111).to(torch.long).reshape(-1)
    val0 = torch.index_select(quant_state.code, dim=0, index=elm0)
    val1 = torch.index_select(quant_state.code, dim=0, index=elm1)

    # blocksize = quant_state.blocksize
    # num_blocks = A.shape[1] // blocksize
    # absmax = absmax.view(num_blocks, 1)
    # val0 = (val0.view(num_blocks, blocksize) * absmax).to(torch.bfloat16).view(-1)
    # val1 = (val1.view(num_blocks, blocksize) * absmax).to(torch.bfloat16).view(-1)

    # TODO: make this more elegant
    blocksize = quant_state.blocksize
    num_blocks = A.shape[1] * 2 // blocksize
    result = torch.stack([val0, val1], dim=-1).view(num_blocks, blocksize)
    absmax = absmax.view(num_blocks, 1)
    return (result * absmax).reshape(quant_state.shape).to(torch.bfloat16).t()

    # return torch.stack([val0, val1], dim=-1).reshape(quant_state.shape)

def print_differences(tensor1, tensor2):
    # Define tolerances (same as torch.allclose defaults)
    rtol = 1e-5  # Relative tolerance
    atol = 1e-8  # Absolute tolerance

    # Compute absolute difference
    abs_diff = torch.abs(tensor1 - tensor2)

    # Identify elements that are NOT close
    not_close = abs_diff > (atol + rtol * torch.abs(tensor2))

    # Get 2D indices of differing elements
    indices = torch.nonzero(not_close, as_tuple=False)  # Shape [num_diffs, 2]

    # Extract differing elements
    diff_elements1 = tensor1[not_close]
    diff_elements2 = tensor2[not_close]

    # Print the results
    print(f"Number of differing elements: {indices.shape[0]}")
    print("Row,Col | Tensor1 Value | Tensor2 Value")
    print("-------------------------------------")
    for idx, val1, val2 in zip(indices, diff_elements1, diff_elements2):
        row, col = idx[0].item(), idx[1].item()
        print(f"{row:3d},{col:3d} | {val1.item():13.6f} | {val2.item():13.6f}")

# https://stackoverflow.com/questions/18425225/getting-the-name-of-a-variable-as-a-string
def NAME(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    names = [var_name for var_name, var_val in callers_local_vars if var_val is var]
    return names[0] if len(names) != 0 else ""

def assert_same(x, y):
    try: torch.testing.assert_close(x, y, check_stride = True)
    except Exception as error:
        raise RuntimeError(
            f"Failed allclose: {NAME(x)}, {NAME(y)}\n{str(error)}"
        )
