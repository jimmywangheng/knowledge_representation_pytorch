# Knowledge Graph Representation Pytorch

## Introduction

We provide code for several knowledge graph representation algorithms here, including TransE, TransH, TransR, and TransD.

Every algorithm has two programs. The program name without "Bernoulli" generates negative samples by corrupting head and tail entities with equal probabilities, otherwise, it generates negative samples by corrupting head and tail entities according to Bernoulli Distribution (See [Knowledge Graph Embedding by Translating on Hyperplanes](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546)).

我们提供了TransE、TransH、TransR和TransD四个知识图谱表示的算法。

每个算法包含两个程序。文件名不带“Bernoulli”的程序，以相同概率更换头实体和尾实体的方式生成负样本；文件名带“Bernoulli”的程序，则依据伯努利分布更换头实体和尾实体，生成负样本。（参见[Knowledge Graph Embedding by Translating on Hyperplanes](https://www.aaai.org/ocs/index.php/AAAI/AAAI14/paper/view/8531/8546)）

## Code Structure

`utils.py` contains basic **Triple** class, its comparison, and other basic computations. It also contains the code for computing head-tail proportion for each relation, classifying relations into 1-1, 1-N, N-1 and N-N, and dividing triples according to them.

`data.py` contains various ways to generate negative triples and get a batch of training samples and its corresponding negative samples.

`model.py` contains our four models used in knowledge graph representation.

`projection.py` contains the projection functions used in TransH, TransR, and TransD.

`loss.py` contains the loss functions used in our algorithms, of which the most important is margin loss and orthogonal loss.

`evaluation.py` is evaluating the model in two metrics: meanrank and hits@10. We use multi-processing to speed up computation.

`utils.py`包含**Triple**基类，其比较操作和其他基本的算术操作，也包含了计算每个关系头尾实体比例、将关系分成1-1、1-N、N-1、N-N等四种、以及按关系种类划分三元组的方法。

`data.py`包含了多种生成负例三元组，以及生成一批次正样本和负样本的方法。

`model.py`包含了我们实现的知识表示的四个模型。

`projection.py`包含了TransH、TransR和TransD的投影函数。

`loss.py`包含了我们在算法中用到的损失函数，最重要的是边界损失函数和正交损失函数。

`evaluation.py`包含了评测模型性能的两个维度：meanrank（平均排名）和hits@10（前10名比率）。我们使用了多进程加速计算。

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

我们的程序都是用Python 3.6编写的，使用PyTorch作为深度学习框架。运行我们的程序之前，请先安装Python和PyTorch。

我们使用了hyperboard实时展示训练曲线和验证曲线。对hyperboard感兴趣的用户，请参考https://github.com/WarBean/hyperboard 。

用法：

python xxx.py [参数]

可以选择的参数如下：

`-d [str]`: 使用哪个数据集？可能的选择有FB13、FB15k、WN11和WN18。

`-l [float]`: 初始学习率，适合于TransE和TransH。默认0.001.

`-l1 [float]`: 第一阶段的学习率。适合TransR和TransD。默认0.001.

`-l2 [float]`: 第二阶段的初始学习率，如果-es参数设置大于0. 适合TransR和TransD。默认0.0005.

`-es [int]`: 减少学习率的次数。若设置为0，则整个过程中学习率保持一致。默认0.

`-L [int]`: 如果设为1，则使用L1作为距离函数，否则使用L2。默认1.

`-em [int]`: 实体和关系向量的维数。默认100.

`-nb [int]`: 每轮训练使用多少批样本。默认100.

`-n [int]`: 最多训练多少轮。默认1,000.

`-m [float]`: 边界损失函数的边界值。默认1.0.

`-f [int]`: 在训练、验证和测试过程中，是否将假负例过滤掉。如果设为1，则过滤。默认1.

`-mo [float]`: 优化器的动量。默认0.9.

`-s [int]`: 固定随机种子。若设为0，则随机种子不固定。默认0.

`-op [int]`: 使用哪种优化器。若设为0，则使用随机梯度下降（SGD）；若设为1，则使用Adam。默认1.

`-p [int]`: hyperboard的端口号。默认5000.

`-np [int]`: 评估过程中开的进程数。默认4.

测试集上的评估结果，将被写入./result/[数据集名称].txt的文本文件中，例如./result/FB15k.txt。结果包括在整个测试集上的Hits@10和MeanRank指标评估结果，以及在1-1、1-N、N-1、N-N子集上分别预测头尾的评估结果。

## Appendices

I'm finding an internship recently. If interested, you can contact me by email: wangh376@mail2.sysu.edu.cn

最近正在找实习，感兴趣的可联系我：wangh376@mail2.sysu.edu.cn


