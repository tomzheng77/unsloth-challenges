import time

from bitsandbytes.nn import Linear4bit, Params4bit
import torch
import triton
import triton.language as tl
from torch.library import custom_op

DEBUG_FLAG = False
@triton.jit
def fused_dequantize_kernel_fp32(
    absmax_ptr,        # the absmax values in uint8
    absmax_code_ptr,   # the code values in float32, each absmax value will point to this initially
    absmax_scale_ptr,  # [0...absmax_block_size) should be scaled by absmax_scale_ptr[0], etc etc
    absmax_offset_ptr, # a scalar to add all the absmax values by
    values_ptr,
    values_code_ptr,
    output_ptr,
    n_elements,
    absmax_block_size: tl.constexpr,
    fraction_values_block_size: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    packed_block_size: tl.constexpr,
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)
    indices = tl.arange(0, packed_block_size)

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

    # assume fraction_values_block_size is 64 / 8 == 8, this means that
    # subgroup_offsets = [0, 0, 0, ... (4 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    subgroup_offsets = indices // fraction_values_block_size

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + indices, subgroup_offsets)

    expanded_absmax_idx = pid * absmax_block_size + subgroup_offsets
    expanded_absmax_idx_quantized = tl.load(absmax_ptr + expanded_absmax_idx).to(tl.int32)
    expanded_absmax_val = tl.load(absmax_code_ptr + expanded_absmax_idx_quantized)
    expanded_absmax_scale = tl.load(absmax_scale_ptr + pid)
    expanded_absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant

    expanded_absmax_intermediate = (expanded_absmax_val * expanded_absmax_scale)

    # TODO remove this flush, it is needed for some unknown reason
    # TODO likely because it influences the PTX
    tl.store(
        output_ptr + pid * TRITON_BLOCK_SIZE + indices,
        indices, # no need for bfloat16 hack
        mask=(indices < 0),
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

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + indices, expanded_absmax_final)

    # now lets just write something, anything to the values
    values_idx = pid * packed_block_size + indices
    values_val_packed = tl.load(values_ptr + values_idx).to(tl.uint32, bitcast=True)

    val0 = (values_val_packed >> 4) & 0b1111
    val1 = (values_val_packed >> 0) & 0b1111
    val2 = (values_val_packed >> 12) & 0b1111
    val3 = (values_val_packed >> 8) & 0b1111
    val4 = (values_val_packed >> 20) & 0b1111
    val5 = (values_val_packed >> 16) & 0b1111
    val6 = (values_val_packed >> 28) & 0b1111
    val7 = (values_val_packed >> 24) & 0b1111

    values_val0 = tl.load(values_code_ptr + val0)
    values_val1 = tl.load(values_code_ptr + val1)
    values_val2 = tl.load(values_code_ptr + val2)
    values_val3 = tl.load(values_code_ptr + val3)
    values_val4 = tl.load(values_code_ptr + val4)
    values_val5 = tl.load(values_code_ptr + val5)
    values_val6 = tl.load(values_code_ptr + val6)
    values_val7 = tl.load(values_code_ptr + val7)

    output_base = output_ptr + pid * TRITON_BLOCK_SIZE
    out_offsets0 = output_base + 8 * indices
    out_offsets1 = out_offsets0 + 1
    out_offsets2 = out_offsets0 + 2
    out_offsets3 = out_offsets0 + 3
    out_offsets4 = out_offsets0 + 4
    out_offsets5 = out_offsets0 + 5
    out_offsets6 = out_offsets0 + 6
    out_offsets7 = out_offsets0 + 7

    # at least this populates everything
    # tensor([[-0.1848, 0.5626, -0.2844, ..., 0.7230, -0.1848, 0.1609],
    #         [0.1609, 0.1609, 0.3379, ..., 0.4407, -1.0000, -0.6962],
    #         [0.2461, 0.5626, -0.0911, ..., -0.6962, 0.5626, -0.6962],
    #         ...,
    #         [0.7230, 0.1609, 0.3379, ..., -0.2844, 0.3379, 0.5626],
    #         [-0.5251, 0.3379, -1.0000, ..., 0.4407, -0.6962, 0.0796],
    #         [0.3379, 0.7230, -0.0911, ..., -0.5251, 0.7230, 0.7230]],
    #        device='cuda:0')
    # tl.store(out_offsets0, pid * packed_block_size + indices)
    # tl.store(out_offsets1, pid * packed_block_size + indices)
    # tl.store(out_offsets0, values_val0)
    # tl.store(out_offsets1, values_val1)
    # tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    # tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    # tl.store(out_offsets0, values_val0 * expanded_absmax_final + expanded_absmax_offset)
    # tl.store(out_offsets1, values_val1 * expanded_absmax_final + expanded_absmax_offset)

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    # subgroup_offsets = indices // half_values_block_size
    #
    # # access absmax_final with subgroup_offsets
    # absmax_final_subgroup = tl.load(absmax_final + subgroup_offsets)

    tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    tl.store(out_offsets2, (values_val2 * expanded_absmax_final))
    tl.store(out_offsets3, (values_val3 * expanded_absmax_final))
    tl.store(out_offsets4, (values_val4 * expanded_absmax_final))
    tl.store(out_offsets5, (values_val5 * expanded_absmax_final))
    tl.store(out_offsets6, (values_val6 * expanded_absmax_final))
    tl.store(out_offsets7, (values_val7 * expanded_absmax_final))

# almost entire duplicate, except with output step changed
@triton.jit
def fused_dequantize_kernel_bfloat16_fp32(
    absmax_ptr,        # the absmax values in uint8
    absmax_code_ptr,   # the code values in float32, each absmax value will point to this initially
    absmax_scale_ptr,  # [0...absmax_block_size) should be scaled by absmax_scale_ptr[0], etc etc
    absmax_offset_ptr, # a scalar to add all the absmax values by
    values_ptr,
    values_code_ptr,
    output_ptr,
    n_elements,
    absmax_block_size: tl.constexpr,
    half_values_block_size: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    packed_block_size: tl.constexpr,
):
    # Program ID: each thread block processes one output block
    pid = tl.program_id(axis=0)
    indices = tl.arange(0, packed_block_size)

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
    subgroup_offsets = indices // half_values_block_size

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + indices, subgroup_offsets)

    expanded_absmax_idx = pid * absmax_block_size + subgroup_offsets
    expanded_absmax_idx_quantized = tl.load(absmax_ptr + expanded_absmax_idx).to(tl.int32)
    expanded_absmax_val = tl.load(absmax_code_ptr + expanded_absmax_idx_quantized)
    expanded_absmax_scale = tl.load(absmax_scale_ptr + pid)
    expanded_absmax_offset = tl.load(absmax_offset_ptr) # TODO maybe use constant

    expanded_absmax_intermediate = (expanded_absmax_val * expanded_absmax_scale)

    # TODO remove this flush, it is needed for some unknown reason
    # TODO likely because it influences the PTX
    tl.store(
        output_ptr + pid * TRITON_BLOCK_SIZE + indices,
        (expanded_absmax_intermediate.to(tl.uint32, bitcast=True) & 0xFFFF).to(tl.uint16).to(tl.bfloat16, bitcast=True),
        mask=(indices < 0),
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

    # tl.store(output_ptr + pid * TRITON_BLOCK_SIZE + indices, expanded_absmax_final)

    # now lets just write something, anything to the values
    values_idx = pid * packed_block_size + indices
    values_val_packed = tl.load(values_ptr + values_idx).to(tl.int32)

    val0 = (values_val_packed >> 4).to(tl.int32)  # High 4 bits
    val1 = (values_val_packed & 0b1111).to(tl.int32)  # Low 4 bits

    values_val0 = tl.load(values_code_ptr + val0)
    values_val1 = tl.load(values_code_ptr + val1)

    output_base = output_ptr + pid * TRITON_BLOCK_SIZE
    out_offsets0 = output_base + 2 * indices
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
    # tl.store(out_offsets0, pid * packed_block_size + indices)
    # tl.store(out_offsets1, pid * packed_block_size + indices)
    # tl.store(out_offsets0, values_val0)
    # tl.store(out_offsets1, values_val1)
    # tl.store(out_offsets0, (values_val0 * expanded_absmax_final))
    # tl.store(out_offsets1, (values_val1 * expanded_absmax_final))
    # tl.store(out_offsets0, values_val0 * expanded_absmax_final + expanded_absmax_offset)
    # tl.store(out_offsets1, values_val1 * expanded_absmax_final + expanded_absmax_offset)

    # assume half_values_block_size is 128, this means that
    # subgroup_offsets = [0, 0, 0, ... (128 times) 1, 1, 1, ... 2, 2, 2, ... 3, 3, 3, ...]
    # subgroup_offsets = indices // half_values_block_size
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

@custom_op("mylib::fused_dequantize_op_fp32", mutates_args=())
def fused_dequantize_op_fp32(
    absmax_ptr: torch.Tensor,
    absmax_code_ptr: torch.Tensor,
    absmax_scale_ptr: torch.Tensor,
    absmax_offset_ptr: torch.Tensor,
    values_ptr: torch.Tensor,
    values_code_ptr: torch.Tensor,
    n_elements: int,
    absmax_block_size: int,
    fraction_values_block_size: int,
    TRITON_BLOCK_SIZE: int,
    packed_block_size: int,
    shape_x: int,
    shape_y: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    output_ptr = torch.empty((shape_x, shape_y), dtype=dtype, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['TRITON_BLOCK_SIZE']),)

    print('n_elements', n_elements)
    print(output_ptr.shape, dtype)

    # TODO: only use the special output if < sm_80
    is_bfloat16 = dtype == torch.bfloat16
    kernel = fused_dequantize_kernel_bfloat16_fp32 if is_bfloat16 else fused_dequantize_kernel_fp32
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
        fraction_values_block_size,
        TRITON_BLOCK_SIZE,
        packed_block_size,
    )

    return output_ptr


# Register a fake implementation
@fused_dequantize_op_fp32.register_fake
def fake_fused_dequantize_op_fp32(
    absmax_ptr: torch.Tensor,
    absmax_code_ptr: torch.Tensor,
    absmax_scale_ptr: torch.Tensor,
    absmax_offset_ptr: torch.Tensor,
    values_ptr: torch.Tensor,
    values_code_ptr: torch.Tensor,
    n_elements: int,
    absmax_block_size: int,
    fraction_values_block_size: int,
    TRITON_BLOCK_SIZE: int,
    packed_block_size: int,
    shape_x: int,
    shape_y: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty((shape_x, shape_y), dtype=dtype, device='cuda')

@torch.compile(fullgraph=False)
def fused_dequantize_fp32(A, quant_state):
    assert(A.dtype == torch.float32)
    n_elements = torch.numel(A) * 8
    absmax_block_size = quant_state.state2.blocksize # e.g. 256 (every 256 abs-maxes have a scaling factor)
    values_block_size = quant_state.blocksize # e.g. 64 (every 64 elements have an absmax)
    fraction_values_block_size = values_block_size >> 3

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

    if DEBUG_FLAG:
        assert(absmax_ptr.dtype == torch.uint8)
        assert(absmax_code_ptr.dtype == torch.float32)
        assert(absmax_ptr.device.type == 'cuda')
        assert(absmax_code_ptr.device.type == 'cuda')
        assert(absmax_offset_ptr.dtype == torch.float32)
        assert(absmax_offset_ptr.device.type == 'cuda')

    # NOTE: surely we want one triton block to handle at least an entire absmax block
    TRITON_BLOCK_SIZE = absmax_block_size * values_block_size
    packed_block_size = TRITON_BLOCK_SIZE >> 3

    output_ptr = fused_dequantize_op_fp32(
        absmax_ptr,
        absmax_code_ptr,
        absmax_scale_ptr,
        absmax_offset_ptr,
        values_ptr,
        values_code_ptr,
        n_elements,
        absmax_block_size,
        fraction_values_block_size,
        TRITON_BLOCK_SIZE,
        packed_block_size,
        quant_state.shape[0],
        quant_state.shape[1],
        quant_state.dtype,
    )

    # print(list(fused_dequantize_kernel_fp32.cache[0].values())[0].asm['ptx'])

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
