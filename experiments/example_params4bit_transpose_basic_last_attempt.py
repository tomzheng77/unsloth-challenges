import torch
import bitsandbytes.nn  # Adjust this import based on the actual source of Params4bit
from typing import Any


# Original Params4bit class (for reference, you don’t need to define this)
# class Params4bit:
#     def __init__(self, data: torch.Tensor, **kwargs):
#         self.data = data
#         self.__dict__.update(kwargs)
#     def some_method(self):
#         return self.data * 2

# Your wrapper class
OriginalParams4bit = bitsandbytes.nn.Params4bit
class Params4bitWrap:
    def __init__(self, data: torch.Tensor, **kwargs):
        # Explicitly store the tensor
        self.data = data if isinstance(data, torch.Tensor) else torch.tensor(data)

        # Create an instance of the original Params4bit
        self._params4bit = OriginalParams4bit(self.data, **kwargs)

        # Copy attributes for compatibility
        for key, value in self._params4bit.__dict__.items():
            if key != "data":
                setattr(self, key, value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._params4bit, name)

    def to_parameter(self):
        return torch.nn.Parameter(self.data)


# Monkey patch: Replace Params4bit with Params4bitWrap
bitsandbytes.nn.Params4bit = Params4bitWrap


# Now any code that imports and uses Params4bit will get Params4bitWrap instead
@torch.compile(fullgraph=True)
def my_function(param):
    return param.data + 999  # Should work with torch.compile


# Example: Simulate third-party code creating Params4bit
from bitsandbytes.nn import Params4bit  # This is now Params4bitWrap

param = Params4bit(torch.randn(5))
result = my_function(param)
print(result.to('cuda'))