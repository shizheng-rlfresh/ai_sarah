# this is the implementation for sparse layer for initialization 
# see https://pytorch.org/docs/stable/index.html for reference

import math
import warnings

import torch
from torch import Tensor

from torch.nn import init

def Sparse_kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = Sparse_calculate_correct_fan(tensor, mode)
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        if tensor.size(1) == 1:
            return tensor.uniform_(-bound, bound)
        elif tensor.size(1) > 1:
            a,b = tensor.shape
            _empty = torch.empty(b,a)
            _empty.uniform_(-bound, bound)
            return tensor.copy_(_empty.t())
    
def Sparse_calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = Sparse_calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
    
def Sparse_calculate_fan_in_and_fan_out(tensor):
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(0)
    num_output_fmaps = tensor.size(1)
    
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].numel()
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out