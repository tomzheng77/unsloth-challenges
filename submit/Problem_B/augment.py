def augment_for_torch_compile():
    # Plan: as part of patching the forward function, it would need to call the patched version of dequantize_4bit,
    # and possibly a version of it that also does transposing
    from bitsandbytes import functional

    if not hasattr(functional, 'original_dequantize_4bit'):
        functional.original_dequantize_4bit = functional.dequantize_4bit

    from kernels import fused_dequantize
    import torch

    def augmented_dequantize_4bit(
        A,  # torch.Tensor
        quant_state=None,  # Optional[QuantState]
        absmax=None,  # Optional[torch.Tensor]
        out=None,  # Optional[torch.Tensor]
        blocksize=64,  # int
        quant_type="fp4",
    ):
        assert (quant_state is not None)
        assert (absmax is None)
        assert (out is None)
        assert (blocksize == 64)
        assert (quant_type == "fp4")

        return fused_dequantize(A, quant_state).t()
        # return functional.original_dequantize_4bit(A, quant_state, absmax, out, blocksize, quant_type)

    functional.dequantize_4bit = augmented_dequantize_4bit

    # Plan: patch the forward function of nn.Linear4bit, such that it can be torch.compile(d)
    # this should mean that the backward pass can also be automatically derived
    # first lets store the original functions before patching, this cell should not be modified
    import bitsandbytes as bnb
    if not hasattr(bnb.nn.Linear4bit, 'original_forward'):
        bnb.nn.Linear4bit.original_forward = bnb.nn.Linear4bit.forward

    from torch.compiler import allow_in_graph
    # If this cell is disabled, then it should show the custom dequantize function working
    # Transposition trick, to avoid having the transposition be done by calling Params4bit.t()
    # NOT transpose in the Linear4bit.forward, BUT secretly transpose in the MatMul4Bit
    # maybe instead of it calling MatMul4bit, it could call a custom implementation of torch Function

    import bitsandbytes.functional as F
    from typing import Optional
    from math import prod
    from bitsandbytes.nn.modules import fix_4bit_weight_quant_state_from_module, Params4bit
    ENABLE_ASSERTIONS = False

    class TransposeBMatMul4Bit(torch.autograd.Function):
        # forward is the same, but we added the fallback for pre-turing GPUs
        # backward is mostly the same, but adds one extra clause (see "elif state.CxB is not None")

        @staticmethod
        def forward(ctx, A, B, out=None, bias=None, quant_state: Optional[F.QuantState] = None):
            # default of pytorch behavior if inputs are empty
            ctx.is_empty = False
            if prod(A.shape) == 0:
                ctx.is_empty = True
                ctx.A = A
                ctx.B = B
                ctx.bias = bias
                B_shape = quant_state.shape
                if A.shape[-1] == B_shape[0]:
                    return torch.empty(A.shape[:-1] + B_shape[1:], dtype=A.dtype, device=A.device)
                else:
                    return torch.empty(A.shape[:-1] + B_shape[:1], dtype=A.dtype, device=A.device)

            # 1. Dequantize
            # 2. MatmulnN
            output = torch.nn.functional.linear(A, F.dequantize_4bit(B, quant_state).to(A.dtype).t(),
                                                bias)  # NOTE the transposition

            # 3. Save state
            ctx.state = quant_state
            ctx.dtype_A, ctx.dtype_B, ctx.dtype_bias = A.dtype, B.dtype, None if bias is None else bias.dtype

            if any(ctx.needs_input_grad[:2]):
                ctx.tensors = (None, B)
            else:
                ctx.tensors = (None, None)

            return output

        @staticmethod
        def backward(ctx, grad_output):
            if ctx.is_empty:
                bias_grad = None if ctx.bias is None else torch.zeros_like(ctx.bias)
                return torch.zeros_like(ctx.A), torch.zeros_like(ctx.B), None, bias_grad, None

            req_gradA, _, _, req_gradBias, _ = ctx.needs_input_grad
            _, B = ctx.tensors

            grad_A, grad_B, grad_bias = None, None, None

            if req_gradBias:
                # compute grad_bias first before changing grad_output dtype
                grad_bias = grad_output.sum(0, dtype=ctx.dtype_bias)

            # not supported by PyTorch. TODO: create work-around
            # if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
            if req_gradA:
                grad_A = torch.matmul(grad_output, F.dequantize_4bit(B, ctx.state).to(
                    grad_output.dtype).t())  # NOTE the transposition

            return grad_A, grad_B, None, grad_bias, None

    @torch.compile(fullgraph=True)
    def get_data_transposed(param: Params4bit):
        if isinstance(param.some_data, Params4bit):
            param = param.some_data
        # assert(False)
        # print(param.some_data)
        # print(type(param.data))  # See what Dynamo sees
        if isinstance(param.some_data, Params4bit):
            raise RuntimeError("Expected some_data NOT to be Params4bit")
        # assert isinstance(param.some_data, torch.Tensor)
        return param.some_data.t()

    def inner_transpose_forward(self, x: torch.Tensor):
        fix_4bit_weight_quant_state_from_module(self)

        # weights are cast automatically as Int8Params, but the bias has to be cast manually
        if self.bias is not None and self.bias.dtype != x.dtype:
            self.bias.data = self.bias.data.to(x.dtype)

        if not self.compute_type_is_set:
            self.set_compute_type(x)
            self.compute_type_is_set = True

        inp_dtype = x.dtype
        if self.compute_dtype is not None:
            x = x.to(self.compute_dtype)

        bias = None if self.bias is None else self.bias.to(self.compute_dtype)

        # return bnb.matmul_4bit(x, self.weight.t(), bias=bias, quant_state=self.weight.quant_state).to(inp_dtype)
        # NOTE: here we pass in the data (nf4-packed weights) directly as Matrix B
        # we assume the GEMV path of bnb.matmul_4bit is never taken
        A = x
        B = get_data_transposed(self.weight)
        out = None
        quant_state = self.weight.quant_state

        # shape is  torch.Size([1, 100, 2048]) torch.Size([2097152, 1]) <reversed object at 0x7e1f00bb5cc0>
        # print('shape is ', A.shape, B.shape, reversed(B.shape))
        if ENABLE_ASSERTIONS:
            assert (B.shape[1] == 1)
        return TransposeBMatMul4Bit.apply(A, B, out, bias, quant_state).to(inp_dtype)

    bnb.nn.Linear4bit.forward = inner_transpose_forward
