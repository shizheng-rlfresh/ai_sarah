# AI-SARAH: Adaptive and Implicit Stochastic Recursive Gradient Methods.
paper (As appeared on TMLR 2/2023): [OpenReview link](https://openreview.net/forum?id=WoXJFsJ6Zw&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))
--------------------------------------------------------------------------------------------------------------------
## content:
### python code of implementing algorithms, conducting experiments, and generating figures in the main paper.
### use default setting to reproduce the results in the main paper
--------------------------------------------------------------------------------------------------------------------
## Directory: (Please see detailed instruction in README.txt under each folder) 

### A. Sparse_Init -- sparse layer implementation in Pytorch (.py)

### B. Run -- algorithms to run (.ipynb)

### C. Plot -- plot generation (.ipynb)

### D. Data -- (empty) default folder to download LIBSVM dataset

### E. Logs -- (empty) default folder to save log files for AI-SARAH vs. fine-tuned algorithms (See Chapter "Numerical Experiment")

### F. AllLogs -- (empty) default folder to save log files for all hyper-parameter runs of the other algorithms (See Chapter "Numerical Experiment")

### G. SenseLogs -- (empty) default folder to save log files for sensitivity analysis (See Chapter "AI-SARAH")
--------------------------------------------------------------------------------------------------------------------
## Note:
### a. Each folder contains a README.md for detailed instruction and description
### b. Default computing envirionment is GPU, but all codes are runable with CPU - see code in '/Run' for details
### c. Running time evaluated in this submission code will depend on user's computing environment. 
--------------------------------------------------------------------------------------------------------------------
## Reproducibility:
### To reproduce the figures in the main paper, please download data in '/Data'
#### and run algorithms based on instructions in '/Run' and save logs file based on instruction in '/Logs', ''AllLogs' and/or '/SenseLogs'.
#### Then, the code included in '/Plot' can reproduce the results and figures shown in the main paper
--------------------------------------------------------------------------------------------------------------------
## Some Highlight:
### a. All algorithms are provided with fine-tuned parameters for each dataset presented in the main paper
### b. All algorithms/dataset/case are run with 10 distinct random seeds
### c. AI-SARAH (Algorithm 2) does not require tuning hyper-parameters. User can choose default value of gamma, i.e., 1/32
####   or use any values in {1/8,1/16,1/32,1/64} (or even smaller ones) based on your preference. 
--------------------------------------------------------------------------------------------------------------------
## Code will be made available online upon publication of the paper.
--------------------------------------------------------------------------------------------------------------------

