# this folder contains the implementation of the sparse layer in Pytorch
--------------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------------
## HIGHLIGHT:
## user can define a one-layer fully connected model with Pytorch COO tensor
## data can be loaded into Scipy CSR format
## CNN and RNN are not implemented at this moment (but in our plan for future work)
--------------------------------------------------------------------------------------------------------------------
## Content:
--------------------------------------------------------------------------------------------------------------------
### 1. sparsedata.py   -- 1.1. load LIBSVM (or any data) into CSR format 
###                       1.2. perform basic operation and collect statistics on dataset
###                       1.3. generate mini-batch and convert into into Pytorch COO format
--------------------------------------------------------------------------------------------------------------------
### 2. sparselinear.py -- Sparse (COO) layer implementation of fully connected layer
--------------------------------------------------------------------------------------------------------------------
### 3. sparsemodule.py -- Sparse (COO) Module implementation of one-FC-layer model
--------------------------------------------------------------------------------------------------------------------
### 4. sparseinit.py   -- Replicate Pytorch initialization from dense module/layer/parameters
###                    -- see https://pytorch.org/docs/stable/index.html for reference
--------------------------------------------------------------------------------------------------------------------
### 5. for AI-SARAH:   -- 5.1. sparselinear_v2.py -- allow in-place operation for model parameters
###                       5.2. sparsemodule_v2.py -- sparse module with version 2 sparse linear layer
--------------------------------------------------------------------------------------------------------------------
