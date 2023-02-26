# Sparse Layer Implementation:
# allow forward and backward pass on fully connected layer and tensor in Pytorch COO format
# version 2 allows for non-leaf variable implementation s.t. AI-SARAH code can perform in-place operation with computing graph

import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from Sparse_Init.sparseinit import *


class SparseLinear():
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(SparseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = torch.zeros([self.in_features,self.out_features],requires_grad=True)
        
        if bias:
            self.bias = torch.zeros(self.out_features,requires_grad=True)
        else:
            self.bias = None
        self.reset_parameters()
                    
    def reset_parameters(self) -> None:
        Sparse_kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = Sparse_calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def forward(self, input: Tensor) -> Tensor:
        if self.bias is not None:
            try:
                if input.sparse_dim() == 2:
                    sparse_or_not = True
                    return torch.sparse.mm(input,self.weight)+self.bias
            except:
                return torch.mm(input,self.weight)+self.bias
        else:
            try:
                if input.sparse_dim() == 2:
                    sparse_or_not = True
                    return torch.sparse.mm(input,self.weight)
            except:
                return torch.mm(input,self.weight)
            
    def to(self,device) -> None:
        if torch.cuda.is_available():
            self.weight = self.weight.cuda()
            if self.bias is not None:
                self.bias = self.bias.cuda()

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )