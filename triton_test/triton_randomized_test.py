import torch
from bitsandbytes.nn import Params4bit
from unsloth.kernels import fast_dequantize

from triton_test.triton_increment import fused_dequantize

print('Commencing 100000 iterations of randomized testing')
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
