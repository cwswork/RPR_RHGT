import random
import numpy as np
import torch

def truncated_normal(size, mean=0, std=0.9999):
    (row, col) = size
    with torch.no_grad():
        tmp = torch.FloatTensor(row*col*2).normal_()
        #tmp = tmp.view(2 * row, col)
        trunc = std * 2
        valid = (tmp < trunc) & (tmp > -trunc)

        tensor = tmp[valid]
        idx = random.sample(range(0, len(tensor)), row * col)
        tensor = tensor[idx].view(row, col)
        tensor.data.mul_(std).add_(mean)
        return tensor


