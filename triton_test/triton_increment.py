import time

from bitsandbytes.nn import Linear4bit, Params4bit
import torch
import triton
import triton.language as tl
from unsloth.kernels import fast_dequantize

DEBUG_FLAG = False
@triton.jit
def fused_dequantize_kernel(
    absmax_ptr,        # the absmax values in uint8
    absmax_code_ptr,   # the code values in float32, each absmax value will point to this initially
    absmax_scale_ptr,  # [0...absmax_block_size) should be scaled by absmax_scale_ptr[0], etc etc
    absmax_offset_ptr, # a scalar to add all the absmax values by
    values_ptr,
    values_code_ptr,
    output_ptr,
    n_elements,
    absmax_block_size: tl.constexpr,
    absmax_block_size_step: tl.constexpr,
    half_values_block_size: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE_step: tl.constexpr,
    packed_block_size: tl.constexpr,
    packed_block_size_step: tl.constexpr, # 1/8 of packed_block_size
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)
    sid = tl.program_id(axis=1)

    # ========== [START REFERENCE ABSMAX IMPLEMENTATION] ==========
    # absmax_idx = pid * absmax_block_size + tl.arange(0, absmax_block_size)
    # absmax_val_quantized = tl.load(absmax_ptr + absmax_idx).to(tl.int32)
    #
    # # write this to output_ptr, this checks out
    # tl.store(output_ptr + absmax_idx, absmax_val_quantized)
    #
    # # select the code values to complete the first step of dequant
    # absmax_val = tl.load(absmax_code_ptr + absmax_val_quantized)
    #
    # # matches index_select
    # # tensor([0.0930, 0.1352, 0.0097, ..., -0.1633, 0.0733, 0.1773], device='cuda:0')
    # # tl.store(output_ptr + absmax_idx, absmax_val)
    #
    # absmax_scale = tl.load(absmax_scale_ptr + pid)
    # absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant
    # absmax_final = absmax_val * absmax_scale + absmax_offset
    #
    # # assuming this is correct
    # # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    # # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    # # tl.store(output_ptr + absmax_idx, absmax_final)
    # ========== [END REFERENCE ABSMAX IMPLEMENTATION] ==========

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    subgroup_offsets = tl.arange(0, packed_block_size_step) // half_values_block_size

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), subgroup_offsets)

    expanded_absmax_idx = pid * absmax_block_size + sid * absmax_block_size_step + subgroup_offsets
    expanded_absmax_idx_quantized = tl.load(absmax_ptr + expanded_absmax_idx).to(tl.int32)
    expanded_absmax_val = tl.load(absmax_code_ptr + expanded_absmax_idx_quantized)
    expanded_absmax_scale = tl.load(absmax_scale_ptr + pid)
    expanded_absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant

    expanded_absmax_intermediate = (expanded_absmax_val * expanded_absmax_scale)

    # TODO remove this flush, it is needed for some unknown reason
    # TODO likely because it influences the PTX
    tl.store(
        output_ptr + pid * TRITON_BLOCK_SIZE + sid * TRITON_BLOCK_SIZE_step + tl.arange(0, packed_block_size_step),
        expanded_absmax_intermediate, # no need for bfloat16 hack
        mask=(tl.arange(0, packed_block_size_step) < 1), # NOTE: after sid, also needs to write a single value
    )

    # use explicit ASM to avoid fusing the add with the mul, which results in a fma, which clobbers precision
    # expanded_absmax_final = expanded_absmax_intermediate + expanded_absmax_offset
    expanded_absmax_final = tl.inline_asm_elementwise(
        asm="add.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[expanded_absmax_intermediate, expanded_absmax_offset],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), expanded_absmax_final)

    # now lets just write something, anything to the values
    values_idx = pid * packed_block_size + sid * packed_block_size_step + tl.arange(0, packed_block_size_step)
    values_val_packed = tl.load(values_ptr + values_idx).to(tl.int32)

    val0 = (values_val_packed >> 4).to(tl.int32)  # High 4 bits
    val1 = (values_val_packed & 0b1111).to(tl.int32)  # Low 4 bits

    values_val0 = tl.load(values_code_ptr + val0)
    values_val1 = tl.load(values_code_ptr + val1)

    output_base = output_ptr + pid * TRITON_BLOCK_SIZE + sid * TRITON_BLOCK_SIZE_step
    out_offsets0 = output_base + 2 * tl.arange(0, packed_block_size_step)
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
    # tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    # tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    # tl.store(out_offsets0, values_val0 * expanded_absmax_final + expanded_absmax_offset)
    # tl.store(out_offsets1, values_val1 * expanded_absmax_final + expanded_absmax_offset)

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    # subgroup_offsets = tl.arange(0, packed_block_size) // half_values_block_size
    #
    # # access absmax_final with subgroup_offsets
    # absmax_final_subgroup = tl.load(absmax_final + subgroup_offsets)

    tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    tl.store(out_offsets1, (values_val1 * expanded_absmax_final))

# almost entire duplicate, except with output step changed
@triton.jit
def fused_dequantize_kernel_bfloat16(
    absmax_ptr,        # the absmax values in uint8
    absmax_code_ptr,   # the code values in float32, each absmax value will point to this initially
    absmax_scale_ptr,  # [0...absmax_block_size) should be scaled by absmax_scale_ptr[0], etc etc
    absmax_offset_ptr, # a scalar to add all the absmax values by
    values_ptr,
    values_code_ptr,
    output_ptr,
    n_elements,
    absmax_block_size: tl.constexpr,
    absmax_block_size_step: tl.constexpr,
    half_values_block_size: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE_step: tl.constexpr,
    packed_block_size: tl.constexpr,
    packed_block_size_step: tl.constexpr, # 1/8 of packed_block_size
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)
    sid = tl.program_id(axis=1)

    # ========== [START REFERENCE ABSMAX IMPLEMENTATION] ==========
    # absmax_idx = pid * absmax_block_size + tl.arange(0, absmax_block_size)
    # absmax_val_quantized = tl.load(absmax_ptr + absmax_idx).to(tl.int32)
    #
    # # write this to output_ptr, this checks out
    # tl.store(output_ptr + absmax_idx, absmax_val_quantized)
    #
    # # select the code values to complete the first step of dequant
    # absmax_val = tl.load(absmax_code_ptr + absmax_val_quantized)
    #
    # # matches index_select
    # # tensor([0.0930, 0.1352, 0.0097, ..., -0.1633, 0.0733, 0.1773], device='cuda:0')
    # # tl.store(output_ptr + absmax_idx, absmax_val)
    #
    # absmax_scale = tl.load(absmax_scale_ptr + pid)
    # absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant
    # absmax_final = absmax_val * absmax_scale + absmax_offset
    #
    # # assuming this is correct
    # # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    # # tensor([0.0220, 0.0221, 0.0212, ..., 0.0221, 0.0210, 0.0220], device='cuda:0')
    # # tl.store(output_ptr + absmax_idx, absmax_final)
    # ========== [END REFERENCE ABSMAX IMPLEMENTATION] ==========

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    subgroup_offsets = tl.arange(0, packed_block_size_step) // half_values_block_size

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), subgroup_offsets)

    expanded_absmax_idx = pid * absmax_block_size + sid * absmax_block_size_step + subgroup_offsets
    expanded_absmax_idx_quantized = tl.load(absmax_ptr + expanded_absmax_idx).to(tl.int32)
    expanded_absmax_val = tl.load(absmax_code_ptr + expanded_absmax_idx_quantized)
    expanded_absmax_scale = tl.load(absmax_scale_ptr + pid)
    expanded_absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant

    expanded_absmax_intermediate = (expanded_absmax_val * expanded_absmax_scale)

    # TODO remove this flush, it is needed for some unknown reason
    # TODO likely because it influences the PTX
    tl.store(
        output_ptr + pid * TRITON_BLOCK_SIZE + sid * TRITON_BLOCK_SIZE_step + tl.arange(0, packed_block_size_step),
        (expanded_absmax_intermediate.to(tl.uint32, bitcast=True) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        mask=(tl.arange(0, packed_block_size_step) < 1), # NOTE: after sid, also needs to write a single value
    )

    # use explicit ASM to avoid fusing the add with the mul, which results in a fma, which clobbers precision
    # expanded_absmax_final = expanded_absmax_intermediate + expanded_absmax_offset
    expanded_absmax_final = tl.inline_asm_elementwise(
        asm="add.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[expanded_absmax_intermediate, expanded_absmax_offset],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + tl.arange(0, packed_block_size), expanded_absmax_final)

    # now lets just write something, anything to the values
    values_idx = pid * packed_block_size + sid * packed_block_size_step + tl.arange(0, packed_block_size_step)
    values_val_packed = tl.load(values_ptr + values_idx).to(tl.int32)

    val0 = (values_val_packed >> 4).to(tl.int32)  # High 4 bits
    val1 = (values_val_packed & 0b1111).to(tl.int32)  # Low 4 bits

    values_val0 = tl.load(values_code_ptr + val0)
    values_val1 = tl.load(values_code_ptr + val1)

    output_base = output_ptr + pid * TRITON_BLOCK_SIZE + sid * TRITON_BLOCK_SIZE_step
    out_offsets0 = output_base + 2 * tl.arange(0, packed_block_size_step)
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
    # tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    # tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    # tl.store(out_offsets0, values_val0 * expanded_absmax_final + expanded_absmax_offset)
    # tl.store(out_offsets1, values_val1 * expanded_absmax_final + expanded_absmax_offset)

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    # subgroup_offsets = tl.arange(0, packed_block_size) // half_values_block_size
    #
    # # access absmax_final with subgroup_offsets
    # absmax_final_subgroup = tl.load(absmax_final + subgroup_offsets)

    # workaround to handle bf16 on Tesla T4
    # tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    # tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    val0_bits = tl.cast((values_val0 * expanded_absmax_final), tl.uint32, bitcast=True)  # Updated to use tl.bitcast
    val0_mantissa_bit = (val0_bits & (1 << 16)) != 0
    val0_round_bit = (val0_bits & (1 << 15)) != 0
    val0_sticky_bits = (val0_bits & ((1 << 15) - 1)) != 0
    val0_should_round_up = (val0_round_bit & val0_sticky_bits) | (val0_round_bit & val0_mantissa_bit)
    val0_bits_adjusted = tl.where(val0_should_round_up, val0_bits + (1 << 16), val0_bits)
    val0_bf16_bits = (val0_bits_adjusted >> 16) & 0xffff # TODO handle Inf and NaN
    tl.store(out_offsets0, val0_bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True))

    val1_bits = tl.cast((values_val1 * expanded_absmax_final), tl.uint32, bitcast=True)  # Updated to use tl.bitcast
    val1_mantissa_bit = (val1_bits & (1 << 16)) != 0
    val1_round_bit = (val1_bits & (1 << 15)) != 0
    val1_sticky_bits = (val1_bits & ((1 << 15) - 1)) != 0
    val1_should_round_up = (val1_round_bit & val1_sticky_bits) | (val1_round_bit & val1_mantissa_bit)
    val1_bits_adjusted = tl.where(val1_should_round_up, val1_bits + (1 << 16), val1_bits)
    val1_bf16_bits = (val1_bits_adjusted >> 16) & 0xffff # TODO handle Inf and NaN
    tl.store(out_offsets1, val1_bf16_bits.to(tl.uint16).to(tl.bfloat16, bitcast=True))


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
    values_code_ptr = quant_state.code
    # output_ptr = torch.empty(absmax_ptr.shape, dtype=torch.float32, device='cuda')
    output_ptr = torch.empty(quant_state.shape, dtype=quant_state.dtype, device='cuda')

    if DEBUG_FLAG:
        assert(absmax_ptr.dtype == torch.uint8)
        assert(absmax_code_ptr.dtype == torch.float32)
        assert(absmax_ptr.device.type == 'cuda')
        assert(absmax_code_ptr.device.type == 'cuda')
        assert(absmax_offset_ptr.dtype == torch.float32)
        assert(absmax_offset_ptr.device.type == 'cuda')
        assert(output_ptr.dtype == quant_state.dtype)
        assert(output_ptr.device.type == 'cuda')

    # NOTE: surely we want one triton block to handle at least an entire absmax block
    TRITON_BLOCK_SIZE = absmax_block_size * values_block_size
    packed_block_size = TRITON_BLOCK_SIZE >> 1
    TRITON_BLOCK_SIZE_step = TRITON_BLOCK_SIZE >> 1
    packed_block_size_step = packed_block_size >> 1
    absmax_block_size_step = absmax_block_size >> 1
    grid = lambda meta: (triton.cdiv(n_elements, meta['TRITON_BLOCK_SIZE']),2)

    # TODO: only use the special output if < sm_80
    is_bfloat16 = quant_state.dtype == torch.bfloat16
    kernel = fused_dequantize_kernel_bfloat16 if is_bfloat16 else fused_dequantize_kernel
    kernel[grid](
        absmax_ptr,
        absmax_code_ptr,
        absmax_scale_ptr,
        absmax_offset_ptr,
        values_ptr,
        values_code_ptr,
        output_ptr,
        n_elements,
        absmax_block_size,
        absmax_block_size_step,
        half_values_block_size,
        TRITON_BLOCK_SIZE,
        TRITON_BLOCK_SIZE_step,
        packed_block_size,
        packed_block_size_step,
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

    return output_ptr

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
    print('========== ANSWER 1 ==========')
    answer = fast_dequantize(weight, weight.quant_state)
    print(answer.dtype)
    print(answer)
    print(answer.shape)
    print('========== ANSWER 2 ==========')
    answer = fast_dequantize(weight, weight.quant_state)
    print(answer.dtype)
    print(answer)
    print(answer.shape)

    print('========== MY ANSWER ==========')
    answer = fused_dequantize(weight.data, weight.quant_state)
    print(answer.dtype)
    print(answer)
    print(answer.shape)
    print('========== END ==========')
    # print(list(fused_dequantize_kernel_bfloat16.cache[0].values())[0].asm['ptx'])

    # exit(0)

    # Local: 2.3127870559692383
    start = time.time()
    for i in range(10000):
        fast_dequantize(weight, weight.quant_state)
    print(time.time() - start)

    # Local: 1.411628007888794
    start = time.time()
    for i in range(10000):
        fused_dequantize(weight.data, weight.quant_state)
    print(time.time() - start)

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

# T4 (tuned)
# 3.6271543502807617
# 3.0006306171417236
