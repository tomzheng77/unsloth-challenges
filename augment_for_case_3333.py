# Example of reading a specific case that was dumped
import torch
import json
import os
from bitsandbytes import functional
from torchao.dtypes import NF4Tensor

from functions import my_dequantize_4bit, print_differences, assert_same

# not possible to use this as drop-in, as it also uses CUDA
# from unsloth.kernels import fast_dequantize

# doesn't exactly take in the tensor and QuantState signature
# also it uses the bnb (CUDA) implementation
# from peft.utils.integrations import dequantize_module_weight as peft_dequantize

DIR_NAME = 'dequantize_4bit_cases'
PREFIX = 'case_'
CASE_INDEX = 3333
CHECK_NF4_TENSOR = False
CHECK_ASSUMPTIONS = True

# step through how NF4Tensor does its dequantize, which has been proven to be torch.compile-able
# what does scaler block size correspond to?
if CHECK_NF4_TENSOR:
    example_tensor = torch.randn(32, 32)
    example_nf4 = NF4Tensor.from_tensor(example_tensor, block_size=16, scaler_block_size=8)
    print(example_nf4)
    print(example_nf4.get_original_weight())
    exit(0)

def check_assumptions(A, quant_state):
    assert(A.dtype == torch.uint8)
    assert(quant_state.quant_type == 'nf4')
    assert(quant_state.nested == True)
    assert(quant_state.blocksize == 64)
    assert(quant_state.dtype == torch.bfloat16)
    assert(quant_state.state2.dtype == torch.float32)
    assert(quant_state.state2.blocksize == 256)

    dequantized = functional.dequantize_4bit(A, quant_state)
    assert(dequantized.dtype == torch.bfloat16)
    assert(A.shape[0] * A.shape[1] * 2 == dequantized.shape[0] * dequantized.shape[1])
    assert(quant_state.absmax.shape[0] * quant_state.blocksize == A.shape[0] * A.shape[1] * 2)
    assert(quant_state.state2.absmax.shape[0] * quant_state.state2.blocksize == quant_state.absmax.shape[0])
    assert(quant_state.state2.absmax.dtype == torch.float32)

    # derive_absmax comparison
    # theirs tensor([0.0543, 0.0639, 0.0458,  ..., 0.0420, 0.0398, 0.0382], device='cuda:0')
    # mine tensor([0.0543, 0.0639, 0.0458,  ..., 0.0420, 0.0398, 0.0382], device='cuda:0')
    # theirs tensor([0.0414, 0.0371, 0.0385,  ..., 0.0885, 0.0897, 0.0606], device='cuda:0')
    # mine tensor([0.0414, 0.0371, 0.0385,  ..., 0.0885, 0.0897, 0.0606], device='cuda:0')
    # my_absmax = derive_absmax(A, quant_state)
    # print('mine', my_absmax)

    my_dequantized = my_dequantize_4bit(A, quant_state)
    assert_same(dequantized, my_dequantized)

    # looks like it passes the test from unsloth?
    # if not torch.allclose(dequantized, my_dequantized, atol=1e-9):
    #     print(dequantized)
    #     print(my_dequantized)
    #     print_differences(dequantized, my_dequantized)
    # assert(torch.allclose(dequantized, my_dequantized, atol=1e-3))

    del dequantized
    del my_dequantized

    # understanding, double quantization is always applied, because nested = True


# loop through the cases and sanity check assumptions on the quant_state
if CHECK_ASSUMPTIONS:
    cases = []
    for file in os.listdir(DIR_NAME):
        if file.endswith('_A.pt'):
            assert(os.path.exists(f'{DIR_NAME}/{file[:-5]}_qs.json'))
            A = torch.load(f'{DIR_NAME}/{file}')
            with open(f'{DIR_NAME}/{file[:-5]}_qs.json', 'r') as f:
                qs_dict = json.load(f)
                for key, value in qs_dict.items():
                    if isinstance(value, str) and value.endswith('.pt'):
                        qs_dict[key] = torch.load(value)
                quant_state = functional.QuantState.from_dict(qs_dict, 'cuda')
            check_assumptions(A, quant_state)
            del quant_state

A = torch.load(f'{DIR_NAME}/{PREFIX}{CASE_INDEX}_A.pt')
with open(f'{DIR_NAME}/{PREFIX}{CASE_INDEX}_qs.json', 'r') as f:
    qs_dict = json.load(f)
    for key, value in qs_dict.items():
        if isinstance(value, str) and value.endswith('.pt'):
            qs_dict[key] = torch.load(value)
    quant_state = functional.QuantState.from_dict(qs_dict, 'cuda')
    print(A, quant_state)
    print(quant_state.as_dict())

    assert(quant_state.quant_type == 'nf4')

    functional.dequantize_4bit(A, quant_state)
