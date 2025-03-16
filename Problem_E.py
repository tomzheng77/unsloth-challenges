import torch
from torch import nn

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
    def forward(ctx, X, linear, labels, forward_function):
        # TODO early exit if there is only one sample or if the number
        # TODO of samples isn't a multiple of two
        n_batch = X.shape[0]
        # X0, X1 = torch.chunk(X, chunks=2, dim=0)
        labels_0, labels_1 = torch.chunk(labels, chunks=2, dim=0)
        with torch.enable_grad():
            X0 = X[:n_batch // 2]
            X1 = X[n_batch // 2:]
            assert X0.requires_grad
            assert X1.requires_grad
            Z0 = forward_function(X0, linear, labels_0)
            Z1 = forward_function(X1, linear, labels_1)
            # at some point, realized need to move the `X0 =` and `X1 =` into this block
            # use this to check if grad is working
            output = ((Z0.to(torch.float64) + Z1.to(torch.float64)) * 0.5).to(torch.float32)
        ctx.save_for_backward(X0, X1, Z0, Z1, linear.weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        X0, X1, Z0, Z1, linear_weight = ctx.saved_tensors

        # Gradient scaling factor from the mean
        grad_scale = grad_output * 0.5

        # print('Z0', Z0)
        # print('X0', X0)

        # Compute gradients w.r.t. X1 from Z0
        grad_X0 = torch.autograd.grad(Z0, X0, grad_outputs=grad_scale, retain_graph=True)[0]

        # Compute gradients w.r.t. X0 from Z1
        grad_X1 = torch.autograd.grad(Z1, X1, grad_outputs=grad_scale, retain_graph=True)[0]

        # grad_linear_weight = (
        #     torch.autograd.grad(Z0, linear_weight, grad_outputs=grad_scale, retain_graph=True)[0] +
        #     torch.autograd.grad(Z1, linear_weight, grad_outputs=grad_scale, retain_graph=True)[0]
        # )[0]

        # print('grad_X0', grad_X0)
        # print('grad_X1', grad_X1)
        # print('grad_linear_weight', grad_linear_weight)

        # Assemble full gradient w.r.t. X
        grad_X = torch.cat([grad_X0, grad_X1], dim=0)

        # Return gradients for all inputs
        return grad_X, None, None, None

if __name__ == '__main__':# run tests to see if the outputs match
    for x in range(100):
        input = torch.randn(4, 8, 2, device="cuda", requires_grad=True)
        linear = nn.Linear(2, 4).to("cuda")
        labels = torch.randint(0, 4, (4, 8), device="cuda")
        expected = transformation_function(input, linear, labels)
        actual = MemoryEfficientLinear.apply(input, linear, labels, transformation_function)
        assert(torch.allclose(expected, actual))

        # now check if the backpropagation calculates the same
        expected.backward()
        gradI_expected = input.grad
        gradW_expected = linear.weight.grad
        gradB_expected = linear.bias.grad

        MemoryEfficientLinear.apply(input, linear, labels, transformation_function).backward()
        gradI_actual = input.grad
        gradW_actual = linear.weight.grad
        gradB_actual = linear.bias.grad

        assert(torch.allclose(gradI_expected, gradI_actual))
        assert(torch.allclose(gradW_expected, gradW_actual))
        assert(torch.allclose(gradB_expected, gradB_actual))
