#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 18:58:47
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def projection_transH(original, norm):
	# numpy version
	return original - np.sum(original * norm, axis=1, keepdims=True) * norm

def projection_transH_pytorch(original, norm):
	return original - torch.sum(original * norm, dim=1, keepdim=True) * norm

def projection_transR_pytorch(original, proj_matrix):
	ent_embedding_size = original.shape[1]
	rel_embedding_size = proj_matrix.shape[1] // ent_embedding_size
	original = original.view(-1, ent_embedding_size, 1)
	proj_matrix = proj_matrix.view(-1, rel_embedding_size, ent_embedding_size)
	return torch.matmul(proj_matrix, original).view(-1, rel_embedding_size)

def projection_transD_pytorch_samesize(entity_embedding, entity_projection, relation_projection):
	return entity_embedding + torch.sum(entity_embedding * entity_projection, dim=1, keepdim=True) * relation_projection
