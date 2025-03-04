import torch
# from bitsandbytes.nn import Params4bit
from torch.compiler import allow_in_graph

class Params4bit(torch.nn.Parameter):
    def __new__(cls, data=None, requires_grad=False, quant_state=None, blocksize=64,
                compress_statistics=True, quant_type="fp4", quant_storage=torch.uint8,
                module=None, bnb_quantized=False):
        if data is None:
            data = torch.empty(0)
        self = torch.Tensor._make_subclass(cls, data, requires_grad)
        self.blocksize = blocksize
        self.compress_statistics = compress_statistics
        self.quant_type = quant_type
        self.quant_state = quant_state
        self.quant_storage = quant_storage
        self.bnb_quantized = bnb_quantized
        self._data = data  # Use a private attribute
        self.module = module
        return self

    # NOTE: if we simply use a differently named property, it won't work
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

@allow_in_graph
def get_data_transposed(param: Params4bit):
    return param.data.t()

# Compile it
compiled_get_data_transposed = torch.compile(get_data_transposed, fullgraph=True)

# Test it
tensor = torch.ones(2, 3)
param = Params4bit(tensor)
print(param)
result = compiled_get_data_transposed(param)
print(result.shape)  # Should be torch.Size([3, 2])
