import torch
from torch import nn


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        '''
        indices:(2,nonzero_num)
        values:(nonzero_num)
        shape:(N,N)
        b:(N,1)
        '''
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)  #   a.shape=(N,N)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)   # =>(N,1)

    @staticmethod
    def backward(ctx, grad_output):
        '''
        ctx:
        grad_output:
        '''
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]  # 列数*行数 + 第二行
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    '''  '''
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

