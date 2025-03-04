import torch
from bitsandbytes import nn

class MyCustomObject(torch.nn.Parameter):
    def __init__(self, tensor_data):
        self.data = tensor_data
        super().__init__()

# Create an instance
tensor = torch.randn(2, 3)
obj = MyCustomObject(tensor)

@torch.compile
def process_object(custom_obj):
    # Operate on the tensor member
    return custom_obj.data.t()  # Transpose the tensor

# Run it
result = process_object(obj)
print(result)  # Should print a [3, 2] tensor
