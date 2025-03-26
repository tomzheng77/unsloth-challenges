import torch
from bitsandbytes.nn import Params4bit
from unsloth.kernels import fast_dequantize

from triton_test.triton_increment import fused_dequantize, fused_dequantize_kernel
from converter_kernel import convert_uint8_to_float32
# from triton_test.triton_increment_fp32 import fused_dequantize_fp32

print('Commencing 100000 iterations of randomized testing')

for i in range(10000):
    torch.manual_seed(i)
    tensor = torch.randn((256, 64), dtype=torch.float16, device='cuda')
    weight_uint8 = Params4bit(tensor, quant_type='nf4', quant_storage=torch.uint8).to("cuda")
    weight_uint8_converted = convert_uint8_to_float32(weight_uint8.data)
    weight = Params4bit(tensor, quant_type='nf4', quant_storage=torch.float32).to("cuda")
    # print(weight_uint8_converted)
    # print(weight.data)
    assert(torch.allclose(weight_uint8_converted, weight.data, atol=1e-9, equal_nan=True))

    # works without absmax calculations - what can possibly introduce the inefficiency?
    # weight.quant_state.offset = torch.tensor(0.0, dtype=torch.float32, device='cuda')
    # weight.quant_state.state2.absmax = torch.ones(weight.quant_state.state2.absmax.shape, dtype=torch.float32, device='cuda')
    # weight.quant_state.absmax = torch.full(weight.quant_state.absmax.shape, 0, dtype=torch.uint8, device='cuda')

    actual = fused_dequantize(weight.data.view(dtype=torch.uint8), weight.quant_state)
    expected = fast_dequantize(weight_uint8, weight_uint8.quant_state)


    if not torch.allclose(actual, expected, atol=1e-9):
        torch.set_printoptions(threshold=torch.inf)
        print(f'Iteration {i} failed')
        print(expected)
        print(actual)
        exit(0)

for i in range(10000):
    torch.manual_seed(i)
    tensor = torch.randn((256, 64), dtype=torch.float16, device='cuda')
    weight = Params4bit(tensor, quant_type='nf4').to("cuda")
    actual = fused_dequantize(weight.data, weight.quant_state)
    expected = fast_dequantize(weight, weight.quant_state)
    torch.set_printoptions(precision=18)
    torch.set_printoptions(threshold=torch.inf)

    if not torch.allclose(actual, expected, atol=1e-9):
        print(f'Iteration {i} failed')
        print(expected)
        print(actual)
        exit(0)
