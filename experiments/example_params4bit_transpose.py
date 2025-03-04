import torch
from bitsandbytes.nn import Params4bit
from torch._dynamo.variables import UserDefinedObjectVariable, ConstantVariable
from torch._dynamo.variables.builder import VariableBuilder

# Custom variable to handle Params4bit
class Params4bitVariable(UserDefinedObjectVariable):
    @staticmethod
    def create(tracer, value):
        return Params4bitVariable(value, tracer)

    def call_method(self, tx, name, args, kwargs):
        if name == "t" and self.value.data is not None:
            # Tell Dynamo that param.data.t() is a tensor operation
            return tx.output.create_proxy(
                'call_method',
                't',
                (ConstantVariable.create(self.value.data),),
                {}
            )
        return super().call_method(tx, name, args, kwargs)

# Register the custom variable
# register_user_defined_object(Params4bit, Params4bitVariable)

# Apply the monkey-patch
original_wrap = VariableBuilder._wrap
def _wrap(self, value):
  if isinstance(value, Params4bit):
    # return Params4bitVariable(value, source=self.output.root
    # print(self, value)
    print('dir(value)', dir(value))
    print('dir(self)', dir(self))
    print('dir(self.tx.output)', dir(self.tx.output))
    print('dir(self.source)', dir(self.source))
    # print('dir(self.tx.output.root)', dir(self.tx.output.root))
    return Params4bitVariable(value, source=self.source)
  return original_wrap(self, value)

VariableBuilder._wrap = _wrap

# Your function
@torch.compile(fullgraph=True)
def get_data_transposed(param: Params4bit):
    return param.t()

# Test it
tensor = torch.ones(2, 3)
param = Params4bit(tensor)
print(param)
result = get_data_transposed(param)
print(result.shape)  # Should be torch.Size([3, 2])
