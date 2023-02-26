# implementation of loading LIBSVM data into Scipy CSR format
# pre-processing data
# load (min-batch) samples into Pytorch COO format


import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from scipy.sparse import csr_matrix
import sys
from collections import OrderedDict
from sklearn.datasets import load_svmlight_file

class SparseData():
    def __init__(self, name,device,trfile=None,tefile=None,file=None,special_data=None,split_ratio=0.75,split_seed=0):
        super(SparseData, self).__init__()
        self.name = name
        self.trfile = trfile
        self.tefile = tefile
        self.file = file
        self.special_data = special_data
        self._trainCSR = None
        self._testCSR = None
        self.default_split = False
        self.split_ratio = split_ratio
        self.split_seed = split_seed
        self.device = device
        self.num_label = None   
            
    # read data files for possible rescaling    
    def read(self):  
        try:
            _raw = load_svmlight_file(self.file) 
            return _raw
        except:
            try:
                _trainCSR = load_svmlight_file(self.trfile)
                _testCSR = load_svmlight_file(self.tefile)
                self.default_split = True
                return _trainCSR, _testCSR
            except:
                raise ValueError('data file(s) not specified well to load data')
                
    
    # load rescaled data and do other processing            
    def load(self,_csr=None,_trainCSR=None,_testCSR=None):
        
        if self.default_split:
            self._trainCSR = _trainCSR
            self._testCSR = _testCSR
        else:
            datasize = _csr[1].shape[0]
            allSamples = [i for i in range(datasize)]
            np.random.seed(self.split_seed) 
            np.random.shuffle(allSamples)
            trSize = int(datasize*self.split_ratio)

            self._trainCSR = (_csr[0][allSamples[:trSize]],_csr[1][allSamples[:trSize]])
            self._testCSR = (_csr[0][allSamples[trSize:]],_csr[1][allSamples[trSize:]])
            
        self._stats()
        self._correct_label() 
                
    # stats of the loaded data        
    def _stats(self):
        
        self.trSize = self._trainCSR[1].shape[0]
        self.teSize = self._testCSR[1].shape[0]
        
        self.num_label = len(set(self._trainCSR[1]))
        if self.num_label < len(set(self._testCSR[1])):
            self.num_label = len(set(self._testCSR[1]))
        if self.num_label == 2:
            self.num_label = 1
            
        self.num_feature = self._trainCSR[0].shape[1]
        if self.num_feature < self._testCSR[0].shape[1]:
            self.num_feature = self._testCSR[0].shape[1]
        
        self.trSparse = 1.0-(self._trainCSR[0].getnnz()/(self.trSize*self.num_feature+0.0))
        self.teSparse = 1.0-(self._testCSR[0].getnnz()/(self.teSize*self.num_feature+0.0))
        
        trfeature = list(set(self._trainCSR[0].indices))
        tefeature = list(set(self._testCSR[0].indices))
        
        self.in_tr_not_te = list(np.setdiff1d(trfeature,tefeature,assume_unique=True))
        self.in_te_not_tr = list(np.setdiff1d(tefeature,trfeature,assume_unique=True))
     
    def _correct_label(self):        
        if len(set(self._trainCSR[1]))==2 and len(set(self._testCSR[1]))==2:
            if {1.0,2.0}==set(self._trainCSR[1]):
                list(self._trainCSR)[1][list(self._trainCSR)[1]==1]=-1.0
                list(self._trainCSR)[1][list(self._trainCSR)[1]==2]=1.0
                
                list(self._testCSR)[1][list(self._testCSR)[1]==1]=-1.0
                list(self._testCSR)[1][list(self._testCSR)[1]==2]=1.0
                
            if {0.0,1.0}==set(self._trainCSR[1]):
                list(self._trainCSR)[1][list(self._trainCSR)[1]==0]=-1.0
                list(self._trainCSR)[1][list(self._trainCSR)[1]==1]=1.0
                
                list(self._testCSR)[1][list(self._testCSR)[1]==0]=-1.0
                list(self._testCSR)[1][list(self._testCSR)[1]==1]=1.0
        else:
            if 0 not in self._trainCSR[1]:
                list(self._trainCSR)[1]-=1.0
                list(self._testCSR)[1]-=1.0
        
    def _makecoo(self,SomeCSR=None):
        
        if SomeCSR is None:
            trlabel = self._trainCSR[1]
            telabel = self._testCSR[1]

            traininput = self._trainCSR[0].tocoo()
            testinput = self._testCSR[0].tocoo()
            
            return traininput,testinput,trlabel,telabel    
        
        else:
            somelabel = SomeCSR[1]
            someinput = SomeCSR[0].tocoo()
            
            return someinput,somelabel
        
    def _TCoo(self,val,ind,mb_label,cuda):
        
        ind = torch.LongTensor(ind)
        val = torch.DoubleTensor(val)
        
        num_sample = mb_label.shape[0]
        
        mb_input = torch.sparse.FloatTensor(ind, val, torch.Size([num_sample,self.num_feature]))
        mb_label = torch.tensor(mb_label,dtype=torch.float64)
        
        if self.num_label > 1:
            mb_label = mb_label.type(torch.long)
        
        if cuda:
            mb_input = mb_input.to(self.device)
            mb_label = mb_label.to(self.device)
            
        return mb_input,mb_label
        
        
    def mb(self,SAMPLES,cuda=True):
        
        mb_CSR = (self._trainCSR[0][SAMPLES],self._trainCSR[1][SAMPLES])
        
        # make numpy coo
        mb_input, mb_label = self._makecoo(mb_CSR)
        
        mb_ind = np.stack((mb_input.row,mb_input.col))
        
        # make torch coo
        mb_input,mb_label = self._TCoo(mb_input.data,mb_ind,mb_label,cuda)
        
        return mb_input.data, mb_label
    
    def temb(self,SAMPLES,cuda=True):
        
        te_CSR = (self._testCSR[0][SAMPLES],self._testCSR[1][SAMPLES])
        
        # make numpy coo
        te_input, te_label = self._makecoo(te_CSR)
        
        te_ind = np.stack((te_input.row,te_input.col))
        
        # make torch coo
        te_input,te_label = self._TCoo(te_input.data,te_ind,te_label,cuda)
        
        return te_input.data, te_label    
    
    def __repr__(self):
        
        return '<object_of_sparsedata\n data: %s\n file: %s\n train file: %s\n test file: %s\n default split: %s\n features: %d\n label: %d\n train samples: %d - sparse: %.4f\n test samples: %d - sparse: %.4f\n train label: %s\n test label: %s\n0>'\
    %(self.name,self.file,self.trfile,self.tefile,self.default_split,\
            self.num_feature,self.num_label,self.trSize,self.trSparse,self.teSize,self.teSparse,set(self._trainCSR[1]),\
            set(self._testCSR[1]))