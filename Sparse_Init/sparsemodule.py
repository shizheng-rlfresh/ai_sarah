# Sparse Module implementation with Sparse layer - see sparselayer.py
# So far, only one-layer model is implemented

import torch
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1) #cpu num
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from Sparse_Init.sparselinear import *



class SparseModule(nn.Module):
    def __init__(self, num_feature, num_label, StrongConvex=False, device='cuda:0'):
        super(SparseModule, self).__init__()
        self.num_feature = num_feature
        self.num_label = num_label
        self.StrongConvex = StrongConvex
        self.device=device
        
    def _formation(self,bias=True):
        
        self.bias = bias
        self.layer = SparseLinear(self.num_feature,self.num_label,bias=self.bias)
        
    def numberOfParameters(self):
        return np.sum([  w.numel() for w  in self.parameters()])
    
    def __repr__(self):
        
        try:
            self.layer
            return 'object_of_SparseModule -->\n Layer: %s\n in_feature: %s, out_feature: %s, bias : %s\n StrongConvex: %s, lambda: %s\n device: %s'\
               %(self.layer,self.num_feature,self.num_label,self.bias,\
                 self.StrongConvex, self.lam, self.device)
        except:
            return 'object_of_SparseModule -->\n'
        
        
class ConvexModel(SparseModule):
    def __init__(self, num_feature, num_label, lam=0.0, bias=True, StrongConvex=False, device='cuda:0'):
        super(ConvexModel, self).__init__(num_feature, num_label, StrongConvex, device)
        self.lam = lam
        self.bias = bias
        if self.num_label > 1:
            self.CE = nn.CrossEntropyLoss()
        
        self._formation(bias=self.bias)
        
    def del_in_te_not_tr(self,in_te_not_tr):
        self.layer.weight.data[in_te_not_tr]=0
        
    def forward(self,x,enable_grad=True):
        
        with torch.set_grad_enabled(enable_grad):
            pred = self.layer(x)
            
        if self.num_label > 1:
            return pred
        else:
            return pred.squeeze(1)
        
    def logloss(self,pred,y,enable_grad=True):
        
        if enable_grad and pred.grad_fn is None:
            raise ValueError('enable grad == True but no grad_fn found in %s'%repr(pred))
        
        with torch.set_grad_enabled(enable_grad):    
            if self.num_label==1:
                loss = torch.log(1.0+torch.exp(-y*pred)).sum()/(y.shape[0]+0.0)
            else:
                loss = self.CE(pred,y)
         
            if self.StrongConvex:
                if self.lam == 0.0:
                    raise ValueError('Strong Convex == True but lambda == %s'%repr(self.lam))
                else:
                    loss+=(self.lam/2.0)*torch.stack([(w**2).sum() for w in self.parameters()]).sum()
                
        return loss
    
    
    
    # Loss and Gradient
    def LossGrad(self,data,eval_BS=2000,sample=None,second_order=False):
                
        if sample is None:
            sample = [i for i in range(data.trSize)]
         
        size = len(sample)
        bulk = size//eval_BS
        leftover = size%eval_BS

        eval_bulk = [[eval_BS*i,eval_BS*(i+1)] for i in range(bulk)]
        if leftover > 0:
            eval_bulk = eval_bulk + [[eval_BS*bulk,size]]

        loss = 0.0
        
        V = [w.data*0.0 for w in self.parameters()]
        
        for i in eval_bulk:
            start=i[0]
            end=i[1]

            loss_scale = (end-start+0.0)/size
            
            x_sample,y_sample = data.mb(sample[start:end])
            
            self.zero_grad()
            
            pred_sample = self.forward(x_sample)

            loss_sample = self.logloss(pred_sample,y_sample)
            
            if second_order:
                loss_grad = torch.autograd.grad(loss_sample*loss_scale,self.parameters(),create_graph=True) 
                V = [vi+li+0.0 for vi,li in zip(V,loss_grad)]
            else:                
                loss_grad = torch.autograd.grad(loss_sample*loss_scale,self.parameters()) 
                V = [vi.data+li.data+0.0 for vi,li in zip(V,loss_grad)]

            loss+=loss_sample.item()*loss_scale
        
        return loss,V
    
    def EvaLoss(self,data,eval_BS=2000,sample=None):
        
        if sample is None:
            sample = [i for i in range(data.trSize)]
         
        size = len(sample)
        bulk = size//eval_BS
        leftover = size%eval_BS

        eval_bulk = [[eval_BS*i,eval_BS*(i+1)] for i in range(bulk)]
        if leftover > 0:
            eval_bulk = eval_bulk + [[eval_BS*bulk,size]]

        loss = 0.0
        
        for i in eval_bulk:
            start=i[0]
            end=i[1]

            loss_scale = (end-start+0.0)/size

            x_sample,y_sample = data.mb(sample[start:end])
            
            self.zero_grad()
            
            pred_sample = self.forward(x_sample,enable_grad=False)

            loss_sample = self.logloss(pred_sample,y_sample,enable_grad=False)
            
            loss+=loss_sample.item()*loss_scale
            
        return loss
    
    
    def ComputeAccuracy(self,data,eval_BS=2000):
        size = data.teSize
        bulk = size//eval_BS
        leftover = size%eval_BS

        eval_bulk = [[eval_BS*i,eval_BS*(i+1)] for i in range(bulk)]
        if leftover > 0:
            eval_bulk = eval_bulk + [[eval_BS*bulk,size]]

        accu = 0.0

        for i in eval_bulk:

            subsamples = list(range(i[0],i[1]))
            x,y = data.temb(subsamples)
            pred = self.forward(x,enable_grad=False)

            if data.num_label > 1:
                y_pred = torch.max(F.softmax(pred,dim=1),dim=1)[1]
                accu+=((y-y_pred)==0).sum().item()
            else:
                accu+=(pred*y>0.0).sum().item()

        accu = accu/size

        return accu