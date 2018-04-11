# Knowledge Graph Representation PyTorch

## Introduction

We provide code for several knowledge graph representation algorithms here, including TransE, TransH, TransR, and TransD.

Every algorithm has two programs. The program name without "Bernoulli" generates negative samples by corrupting head and tail entities with equal probabilities, otherwise, it generates negative samples by corrupting head and tail entities according to Bernoulli Distribution (See [Knowledge Graph Embedding by Translating on Hyperplanes](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546)).

## Code Structure

`utils.py` contains basic **Triple** class, its comparison, and other basic computations. It also contains the code for computing head-tail proportion for each relation, classifying relations into 1-1, 1-N, N-1 and N-N, and dividing triples according to them.

`data.py` contains various ways to generate negative triples and get a batch of training samples and its corresponding negative samples.

`model.py` contains our four models used in knowledge graph representation.

`projection.py` contains the projection functions used in TransH, TransR, and TransD.

`loss.py` contains the loss functions used in our algorithms, of which the most important is margin loss and orthogonal loss.

`evaluation.py` is evaluating the model in two metrics: meanrank and hits@10. We use multi-processing to speed up computation.

## Usage

Our programs are all written in Python 3.6, using PyTorch as deep learning architecture. So, please install Python and PyTorch first to run our programs.

We also use hyperboard to display training and validation curve in real time. Users who are interested in hyperboard, please refer to https://github.com/WarBean/hyperboard.

Usage:
python xxx.py [parameters]

Possible parameters includes:

`-d [str]`: Which dataset to use? Possible selections are FB13, FB15k, WN11, WN18.

`-l [float]`: Initial learning rate. Suitable for TransE and TransH. Default 0.001.

`-l1 [float]`: Learning rate for the first phase. Suitable for TransR and TransD. Default 0.001.

`-l2 [float]`: Initial learning rate for the second phase, if -es set to > 0. Suitable for TransR and TransD. Default 0.0005.

`-es [int]`: Number of times for decrease of learning rate. If set to 0, no learning rate decrease will occur. Default 0.

`-L [int]`: If set to 1, it will use L1 as dissimilarity, otherwise L2. Default 1.

`-em [int]`: Embedding size of entities and relations. Default 100.

`-nb [int]`: How many batches to train in one epoch. Default 100.

`-n [int]`: Maximum number of epochs to train. Default 1,000.

`-m [float]`: Margin of margin loss. Default 1.0.

`-f [int]`: Whether to filter false negative triples in training, validating and testing. If set to 1, they will be filtered. Default 1.

`-mo [float]`: Momentum of optimizers. Default 0.9.

`-s [int]`: Fix the random seed, except for 0, which means no random seed is fixed. Default 0.

`-op [int]`: Which optimizer to choose. If set to 0, Stochastic Gradient Descent (SGD) will be used. If set to 1, Adam will be used. Default 1.

`-p [int]`: Port number used by hyperboard. Default 5000.

`-np [int]`: Number of processes when evaluating. Default 4. 

Evaluation result on the test set will be written into ./result/[dataset].txt, such as ./result/FB15k.txt. It consists of Hits@10 and MeanRank on the whole test set, as well as Hits@10 and MeanRank on 1-1, 1-N, N-1, N-N subsets, when predicting head and tail, respectively.
