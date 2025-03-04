import torch

class AnotherInnerObject:
    def __init__(self, tensor_data):
        self.data = tensor_data  # This will be a tensor
        self.other = "some string"  # Non

class MyCustomObject:
    def __init__(self, tensor_data):
        self.data = AnotherInnerObject(tensor_data)  # This will be a tensor
        self.other = "some string"  # Non-tensor member

# Create an instance
tensor = torch.randn(2, 3)
obj = MyCustomObject(tensor)

@torch.compile
def process_object(custom_obj):
    # Operate on the tensor member
    return custom_obj.data.data.t()  # Transpose the tensor

# Run it
result = process_object(obj)
print(result)  # Should print a [3, 2] tensor
