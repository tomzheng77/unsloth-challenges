import torch
from torch import nn

# use memory efficient linear for unsloth GRPO
import os
import gc

os.environ['UNSLOTH_IS_PRESENT'] = '1'
from unsloth_zoo.rl_replacements import UnslothEfficientGRPO

def transformation_function(batch, linear, labels):
    assert batch.requires_grad, "Batch lacks requires_grad"
    x = linear(batch).float()
    assert x.requires_grad, "x lacks requires_grad after linear"
    from torch.nn import CrossEntropyLoss
    down_projection_function = CrossEntropyLoss(reduction = "mean")
    # Down projection to small space
    loss = down_projection_function(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss


# to be very clear about the terminology here
# X is the input to the memory efficient linear function
# Y is W @ X + b, where W is the weight matrix and b is the bias of the linear layer
# Z is f(Y), where f is the transformation function
# the output is expected to be a single scalar value

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    # def forward(ctx, X, linear, labels, forward_function):
    def forward(ctx, *args):
        has_custom_chunk_size = False
        chunk_size = 2
        if isinstance(args[0], int):
            chunk_size = args[0]
            args = args[1:]
            has_custom_chunk_size = True

        X = args[0]
        forward_function = args[-1]
        other_params = args[1:-1]

        # TODO early exit if there is only one sample or if the number
        # TODO of samples isn't a multiple of two
        n_batch = X.shape[0]

        # chunk all other tensors instead of Parameters and Modules
        other_params_chunks = []
        for i in range(len(other_params)):
            other_param = other_params[i]
            assert not isinstance(other_param, list) # to prevent confusing downstream logic
            if isinstance(other_param, torch.Tensor) and not isinstance(other_param, nn.Parameter):
                other_param = list(torch.split(other_param, split_size_or_sections=chunk_size, dim=0))
            other_params_chunks.append(other_param)

        Z_output = None
        with torch.enable_grad():
            X_chunks = torch.split(X, split_size_or_sections=chunk_size, dim=0)
            tensors_for_backward: list[torch.Tensor] = list(X_chunks)
            for i in range(len(X_chunks)):
                X_chunk = X_chunks[i]
                assert(X_chunk.requires_grad)
                size_chunk = X_chunk.shape[0]
                other_params_chunk = []
                for j in range(len(other_params)):
                    other_param = other_params_chunks[j]
                    if isinstance(other_param, list):
                        other_param = other_param[i]
                    other_params_chunk.append(other_param)

                Z_chunk_or_tuple = forward_function(X_chunk, *other_params_chunk)
                if isinstance(Z_chunk_or_tuple, tuple):
                    Z_chunk_or_tuple = Z_chunk_or_tuple[0]

                tensors_for_backward.append(Z_chunk_or_tuple)
                if Z_output is None:
                    Z_output = Z_chunk_or_tuple * size_chunk
                else:
                    Z_output += Z_chunk_or_tuple * size_chunk

            ctx.save_for_backward(*tensors_for_backward)
            ctx.num_args = len(args)
            ctx.has_custom_chunk_size = has_custom_chunk_size
            return (Z_output / n_batch).to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        chunk_count = len(ctx.saved_tensors) // 2
        Xs = ctx.saved_tensors[:chunk_count]
        Zs = ctx.saved_tensors[chunk_count:]

        total_batch_size = sum([X.shape[0] for X in Xs])
        grad_Xs = []
        for i in range(chunk_count):
            X = Xs[i]
            Z = Zs[i]
            batch_size = X.shape[0]
            # grad_scale = grad_output * (batch_size / total_batch_size)
            # TODO understand why passing grad_outputs=grad_scale doesn't work
            grad_X = torch.autograd.grad(Z, X, retain_graph=True)[0]
            grad_Xs.append(grad_X * (batch_size / total_batch_size))

        # Assemble full gradient w.r.t. X
        grad_X = torch.cat(grad_Xs, dim=0)
        padding = [None] if ctx.has_custom_chunk_size else []
        return tuple(padding + [grad_X if i == 0 else None for i in range(ctx.num_args)])

if __name__ == '__main__':# run tests to see if the outputs match
    for x in range(2):
        input_original = torch.randn(4, 8, 2, device="cuda", requires_grad=True)
        linear = nn.Linear(2, 4).to("cuda")
        labels = torch.randint(0, 4, (4, 8), device="cuda")

        input = input_original.clone().detach().requires_grad_(True)
        expected = transformation_function(input, linear, labels)

        expected.backward()
        gradI_expected = torch.clone(input.grad)
        gradW_expected = torch.clone(linear.weight.grad)
        gradB_expected = torch.clone(linear.bias.grad)

        input = input_original.clone().detach().requires_grad_(True)
        actual = MemoryEfficientLinear.apply(input, linear, labels, transformation_function)
        assert(torch.allclose(expected, actual))

        # now check if the backpropagation calculates the same
        actual.backward()
        gradI_actual = input.grad
        gradW_actual = linear.weight.grad
        gradB_actual = linear.bias.grad

        assert(torch.allclose(gradI_expected, gradI_actual))
        assert(torch.allclose(gradW_expected, gradW_actual))
        assert(torch.allclose(gradB_expected, gradB_actual))

    for chunk_size in [1, 2, 4]:
        print(f"Testing chunk size {chunk_size}")
        for i in range(10):
            gc.collect()
            torch.cuda.empty_cache()

            batch_size = 8
            old_hidden_states_original = torch.randn(batch_size, 241, 2048, dtype=torch.bfloat16, device="cuda", requires_grad=False)
            new_hidden_states_original = old_hidden_states_original.clone().detach().requires_grad_(True)  # don't deviate too much to blow out K-L divergence
            lm_head_original = torch.randn(128256, 2048, dtype=torch.bfloat16, device="cuda", requires_grad=True)
            labels = torch.randint(0, 128256, (batch_size, 240), dtype=torch.int64, device="cuda")
            completion_input_ids = torch.randint(0, 128256, (batch_size, 240), dtype=torch.int64, device="cuda")

            # filter out 128004
            # completion_mask = torch.randint(0, 2, (6, 240), dtype=torch.int64, device="cuda")
            completion_mask = torch.ones_like(completion_input_ids)
            advantages_original = torch.randn(batch_size, dtype=torch.float32, device="cuda")
            # advantages = torch.zeros(6, dtype=torch.float32, device="cuda")
            beta = 0.04
            scaler = None
            n_chunks = 1

            old_hidden_states = old_hidden_states_original.clone().detach()
            new_hidden_states = new_hidden_states_original.clone().detach().requires_grad_(True)
            lm_head = nn.Parameter(lm_head_original.clone().detach().requires_grad_(True))
            advantages = advantages_original.clone().detach().requires_grad_(False)
            result = UnslothEfficientGRPO.apply(
                new_hidden_states,
                old_hidden_states,
                lm_head,
                completion_input_ids,
                completion_mask,
                advantages,
                beta,
                scaler,
                n_chunks,
            )[0]

            result.backward()
            expected_grad = new_hidden_states.grad
            # print(result)
            # print(new_hidden_states.grad)

            old_hidden_states = old_hidden_states_original.clone().detach()
            new_hidden_states = new_hidden_states_original.clone().detach().requires_grad_(True)
            lm_head = nn.Parameter(lm_head_original.clone().detach().requires_grad_(True))
            advantages = advantages_original.clone().detach().requires_grad_(False)
            reiter = MemoryEfficientLinear.apply(
                chunk_size,
                new_hidden_states,
                old_hidden_states,
                lm_head,
                completion_input_ids,
                completion_mask,
                advantages,
                beta,
                scaler,
                UnslothEfficientGRPO.apply,
            )

            reiter.backward()
            actual_grad = new_hidden_states.grad
            # print(reiter)
            # print(actual_grad)
            assert(torch.allclose(reiter, result))
            assert(torch.allclose(actual_grad, expected_grad))
