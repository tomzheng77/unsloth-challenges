import time

from PIL.ImageChops import offset
from bitsandbytes.nn import Linear4bit, Params4bit
import torch
import triton
import triton.language as tl
from mpmath import absmax
from unsloth.kernels import fast_dequantize

from functions import my_dequantize_4bit
from peft.utils.integrations import dequantize_module_weight as peft_dequantize

DEBUG_FLAG = False
@triton.jit
def obtain_absmax_kernel(
    absmax_ptr,        # the absmax values in uint8
    absmax_code_ptr,   # the code values in float32, each absmax value will point to this initially
    absmax_scale_ptr,  # [0...absmax_block_size) should be scaled by absmax_scale_ptr[0], etc etc
    absmax_offset_ptr, # a scalar to add all the absmax values by
    values_ptr,
    values_code_ptr,
    output_ptr,
    absmax_output_ptr,
    expanded_absmax_output_ptr,
    n_elements,
    absmax_block_size: tl.constexpr,
    half_values_block_size: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    packed_block_size: tl.constexpr,
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)

    # ========== [START REFERENCE ABSMAX IMPLEMENTATION] ==========
    absmax_idx = pid * absmax_block_size + tl.arange(0, absmax_block_size)
    absmax_val_quantized = tl.load(absmax_ptr + absmax_idx).to(tl.int32)

    # write this to output_ptr, this checks out
    # tl.store(output_ptr + absmax_idx, absmax_val_quantized)

    # select the code values to complete the first step of dequant
    absmax_val = tl.load(absmax_code_ptr + absmax_val_quantized).to(tl.float32)

    # matches index_select
    # tensor([0.0930, 0.1352, 0.0097, ..., -0.1633, 0.0733, 0.1773], device='cuda:0')
    # tl.store(output_ptr + absmax_idx, absmax_val)

    absmax_scale = tl.load(absmax_scale_ptr + pid).to(tl.float32)
    absmax_offset = tl.load(absmax_offset_ptr).to(tl.float32) # TODO maybe use constant
    absmax_final = (absmax_val * absmax_scale).to(tl.float32) + absmax_offset

    # assuming this is correct
    # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    tl.store(absmax_output_ptr + absmax_idx, absmax_final)
    # ========== [END REFERENCE ABSMAX IMPLEMENTATION] ==========

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    subgroup_offsets = tl.arange(0, packed_block_size) // half_values_block_size

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), subgroup_offsets)

    expanded_absmax_idx = pid * absmax_block_size + subgroup_offsets
    expanded_absmax_idx_quantized = tl.load(absmax_ptr + expanded_absmax_idx).to(tl.int32)
    expanded_absmax_val = tl.load(absmax_code_ptr + expanded_absmax_idx_quantized).to(tl.float32)
    expanded_absmax_scale = tl.load(absmax_scale_ptr + pid).to(tl.float32)
    expanded_absmax_offset = tl.load(absmax_offset_ptr).to(tl.float32) # TODO maybe use constant
    expanded_absmax_final = expanded_absmax_val * expanded_absmax_scale + expanded_absmax_offset

    tl.store(expanded_absmax_output_ptr + tl.arange(0, packed_block_size), expanded_absmax_final)

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), expanded_absmax_final)

    # now lets just write something, anything to the values
    values_idx = pid * packed_block_size + tl.arange(0, packed_block_size)
    values_val_packed = tl.load(values_ptr + values_idx).to(tl.int32)

    val0 = (values_val_packed >> 4).to(tl.int32)  # High 4 bits
    val1 = (values_val_packed & 0b1111).to(tl.int32)  # Low 4 bits

    values_val0 = tl.load(values_code_ptr + val0)
    values_val1 = tl.load(values_code_ptr + val1)

    output_base = output_ptr + pid * TRITON_BLOCK_SIZE
    out_offsets0 = output_base + 2 * tl.arange(0, packed_block_size)
    out_offsets1 = out_offsets0 + 1

    # at least this populates everything
    # tensor([[-0.1848, 0.5626, -0.2844, ..., 0.7230, -0.1848, 0.1609],
    #         [0.1609, 0.1609, 0.3379, ..., 0.4407, -1.0000, -0.6962],
    #         [0.2461, 0.5626, -0.0911, ..., -0.6962, 0.5626, -0.6962],
    #         ...,
    #         [0.7230, 0.1609, 0.3379, ..., -0.2844, 0.3379, 0.5626],
    #         [-0.5251, 0.3379, -1.0000, ..., 0.4407, -0.6962, 0.0796],
    #         [0.3379, 0.7230, -0.0911, ..., -0.5251, 0.7230, 0.7230]],
    #        device='cuda:0')
    # tl.store(out_offsets0, pid * packed_block_size + tl.arange(0, packed_block_size))
    # tl.store(out_offsets1, pid * packed_block_size + tl.arange(0, packed_block_size))
    # tl.store(out_offsets0, values_val0)
    # tl.store(out_offsets1, values_val1)
    tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    # tl.store(out_offsets0, values_val0 * expanded_absmax_final + expanded_absmax_offset)
    # tl.store(out_offsets1, values_val1 * expanded_absmax_final + expanded_absmax_offset)

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    # subgroup_offsets = tl.arange(0, packed_block_size) // half_values_block_size
    #
    # # access absmax_final with subgroup_offsets
    # absmax_final_subgroup = tl.load(absmax_final + subgroup_offsets)


def fused_dequantize(A, quant_state):
    n_elements = torch.numel(A) * 2
    absmax_block_size = quant_state.state2.blocksize # e.g. 256 (every 256 abs-maxes have a scaling factor)
    values_block_size = quant_state.blocksize # e.g. 64 (every 64 elements have an absmax)
    half_values_block_size = values_block_size >> 1

    if DEBUG_FLAG:
        print(f'{n_elements=}')
        print(f'{absmax_block_size=}')
        print(f'{values_block_size=}')

    absmax_ptr = quant_state.absmax
    absmax_code_ptr = quant_state.state2.code
    absmax_scale_ptr = quant_state.state2.absmax
    absmax_offset_ptr = quant_state.offset
    values_ptr = A
    # values_code_ptr = quant_state.code
    # output_ptr = torch.empty(absmax_ptr.shape, dtype=torch.float32, device='cuda')
    output_ptr = torch.empty(quant_state.shape, dtype=quant_state.dtype, device='cuda')
    absmax_output_ptr = torch.empty(absmax_ptr.shape, dtype=torch.float32, device='cuda')

    values_code_ptr = torch.tensor([
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ], dtype=torch.float32, device='cuda')

    if DEBUG_FLAG:
        assert(absmax_ptr.dtype == torch.uint8)
        assert(absmax_ptr.device.type == 'cuda')
        assert(absmax_code_ptr.dtype == torch.float32)
        assert(absmax_code_ptr.device.type == 'cuda')
        assert(absmax_scale_ptr.dtype == torch.float32)
        assert(absmax_scale_ptr.device.type == 'cuda')
        assert(absmax_offset_ptr.dtype == torch.float32)
        assert(absmax_offset_ptr.device.type == 'cuda')
        assert(output_ptr.dtype == quant_state.dtype)
        assert(output_ptr.device.type == 'cuda')

    # NOTE: surely we want one triton block to handle at least an entire absmax block
    TRITON_BLOCK_SIZE = absmax_block_size * values_block_size
    packed_block_size = TRITON_BLOCK_SIZE >> 1
    expanded_absmax_output_ptr = torch.empty(packed_block_size, dtype=torch.float32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['TRITON_BLOCK_SIZE']),)
    obtain_absmax_kernel[grid](
        absmax_ptr,
        absmax_code_ptr,
        absmax_scale_ptr,
        absmax_offset_ptr,
        values_ptr,
        values_code_ptr,
        output_ptr,
        absmax_output_ptr,
        expanded_absmax_output_ptr,
        n_elements,
        absmax_block_size,
        half_values_block_size,
        TRITON_BLOCK_SIZE,
        packed_block_size,
    )

    if DEBUG_FLAG:
        absmax_selected = torch.index_select(absmax_code_ptr, 0, absmax_ptr.to(torch.int32)).to(torch.float32)
        num_absmax_blocks = absmax_scale_ptr.numel()
        large_tensor_reshaped = absmax_selected.view(num_absmax_blocks, absmax_block_size)
        small_tensor_reshaped = absmax_scale_ptr.view(num_absmax_blocks, 1)
        absmax_scaled = ((large_tensor_reshaped * small_tensor_reshaped).reshape(-1)) + absmax_offset_ptr

        print(output_ptr.shape)
        print(absmax_ptr)
        print(absmax_selected)
        print(absmax_scaled)
        print('========== OUTPUT ==========')
        print(output_ptr)

    return (output_ptr, absmax_output_ptr, expanded_absmax_output_ptr)

def bnb_Linear4bit(hd, m, dtype = torch.float16):
    return Linear4bit(
        hd, m, bias = None,
        compute_dtype       = dtype,
        compress_statistics = True,
        quant_type          = "nf4",
    )

tensor = torch.randn((2048, 8192), dtype=torch.float16, device='cuda')
weight = Params4bit(tensor, quant_type='nf4').to("cuda")
assert(weight.quant_state.offset.dtype == torch.float32)
assert(weight.quant_state.state2.absmax.dtype == torch.float32)
assert(weight.quant_state.absmax.dtype == torch.uint8)
# weight.quant_state.offset = torch.tensor(0.0, dtype=torch.float32, device='cuda')
# weight.quant_state.state2.absmax = torch.ones(weight.quant_state.state2.absmax.shape, dtype=torch.float32, device='cuda')
# weight.quant_state.absmax = torch.full(weight.quant_state.absmax.shape, 255, dtype=torch.uint8, device='cuda')

# layer = bnb_Linear4bit(2048, 8192).to("cuda")
# weight = layer.weight
# weight_clone = weight.to("cpu").to("cuda")
# assert(weight.quant_state.offset.dtype == torch.float32)
# assert(weight.quant_state.state2.absmax.dtype == torch.float32)
# assert(weight.quant_state.absmax.dtype == torch.uint8)
# weight.quant_state.offset = torch.tensor(0.0, dtype=torch.float32, device='cuda')
# weight.quant_state.state2.absmax = torch.ones(weight.quant_state.state2.absmax.shape, dtype=torch.float32, device='cuda')
# weight.quant_state.absmax = torch.full(weight.quant_state.absmax.shape, 255, dtype=torch.uint8, device='cuda')

# answer = peft_dequantize(weight)
# print(answer.dtype)
# print(answer)
# print(peft_dequantize(layer))
# print(peft_dequantize(layer))

if __name__ == '__main__':
    print('Commencing 100000 iterations of randomized testing')
    for i in range(10000):
        # if i != 159: continue;

        torch.manual_seed(i)
        tensor = torch.randn((256, 64), dtype=torch.float16, device='cuda')
        weight = Params4bit(tensor, quant_type='nf4').to("cuda")
        actual, actual_absmax, actual_absmax_expanded = fused_dequantize(weight.data, weight.quant_state)
        expected, expected_absmax = fast_dequantize(weight, weight.quant_state)

        actual_absmax_expanded = actual_absmax_expanded.view(256, 32)
        torch.set_printoptions(threshold=torch.inf)
        # print(actual_absmax)
        # print(actual_absmax_expanded)

        for i in range(256):
            # print(actual_absmax[i])
            # print(actual_absmax_expanded[i, :])
            if actual_absmax[i] != expected_absmax[i]:
                # actual_absmax[i] tensor(3.127191781997680664062500000000, device='cuda:0')
                # expected_absmax[i] tensor(3.127192020416259765625000000000, device='cuda:0')
                # hand:  3.127191895422129
                # handf32: 3.127192020416259765625000000000
                torch.set_printoptions(precision=30)
                print(f'Iteration {i} failed')
                print('actual_absmax[i]', actual_absmax[i])
                print('expected_absmax[i]', expected_absmax[i])
                print('weight.quant_state.absmax[i]', weight.quant_state.absmax[i], weight.quant_state.absmax[i].dtype)
                print('weight.quant_state.state2.absmax', weight.quant_state.state2.absmax, weight.quant_state.state2.absmax.dtype)
                print('weight.quant_state.state2.code[206]', weight.quant_state.state2.code[206], weight.quant_state.state2.code[206].dtype)
                print('weight.quant_state.offset', weight.quant_state.offset, weight.quant_state.offset.dtype)
            assert(actual_absmax[i] == expected_absmax[i])
            assert torch.all(actual_absmax_expanded[i, :] == actual_absmax[i])

        # exit(0)
        # torch.set_printoptions(threshold=torch.inf)
        # print(weight.quant_state.code)
        # print(weight.quant_state.dtype)
        # exit(0)

        # torch.testing.assert_close(actual, expected, check_stride=True)
        # Iteration 9999 passed for absmax
        assert(torch.allclose(actual_absmax, expected_absmax, atol=1e-18))
        torch.set_printoptions(precision=18)

        # if an absmax has value 2.9456, then it would fail it seems
        compare = torch.tensor(2.9456, device='cuda')
        if not torch.allclose(actual, expected, atol=1e-9):
            torch.set_printoptions(threshold=torch.inf)
            print(f'Iteration {i} failed')
            for row in range(actual.shape[0]):
                has_fails = False
                for col in range(actual.shape[1]):
                    if actual[row, col] != expected[row, col]:
                        print(f'Row {row}, Col {col}')
                        print('actual absmax: ', actual_absmax[row], actual_absmax[row].allclose(compare, atol=1e-9))
                        print('expected absmax: ', expected_absmax[row])
                        print(tensor[row, col])
                        print(actual[row, col])
                        print(expected[row, col])
                        has_fails = True

                if actual_absmax[row].allclose(compare, atol=1e-9) and not has_fails:
                    raise Exception('Found the culprit')
            # print(tensor)
            # print(actual)
            # print(expected)
        # assert(torch.allclose(actual, expected, atol=1e-9))
        # print(f'Iteration {i} passed')

    # print('========== ANSWER 1 ==========')
    # answer = fast_dequantize(weight, weight.quant_state)
    # print(answer.dtype)
    # print(answer)
    # print(answer.shape)
    # print('========== ANSWER 2 ==========')
    # answer = fast_dequantize(weight, weight.quant_state)
    # print(answer.dtype)
    # print(answer)
    # print(answer.shape)
    #
    # print('========== MY ANSWER ==========')
    # answer = fused_dequantize(weight.data, weight.quant_state)
    # print(answer.dtype)
    # print(answer)
    # print(answer.shape)
    # print('========== END ==========')

    # exit(0)

    # Local: 2.3127870559692383
    # start = time.time()
    # for i in range(10000):
    #     fast_dequantize(weight, weight.quant_state)
    # print(time.time() - start)
    #
    # # Local: 1.411628007888794
    # start = time.time()
    # for i in range(10000):
    #     fused_dequantize(weight.data, weight.quant_state)
    # print(time.time() - start)

    # NOTE: my original implementation is flaky, you need to transpose the data
    # even so it can't be directly used in A
    # print(my_dequantize_4bit(weight.data.t(), weight.quant_state))

# L4
# 1.690812587738037
# 1.0677266120910645

# A100
# 0.8383045196533203
# 0.45516395568847656

# T4
# 3.0163562297821045
# 2.995758056640625
