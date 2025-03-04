import torch
from bitsandbytes.nn import Params4bit
from torch.compiler import allow_in_graph

# Monkey-patch to add a property
def ze_data(self) -> torch.Tensor:
    return self.data

Params4bit.ze_data = property(ze_data)

@torch.compile(fullgraph=True)
def get_data_transposed(param: Params4bit):
    return param.ze_data.t()

# Compile it
compiled_get_data_transposed = torch.compile(get_data_transposed, fullgraph=True)

# Test it
tensor = torch.ones(2, 3)
param = Params4bit(tensor)
print(param)
result = get_data_transposed(param)
print(result.shape)  # Should be torch.Size([3, 2])
