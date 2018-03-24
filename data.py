#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 15:31:32
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import random
from copy import deepcopy

from utils import Triple

# Change the head of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_head_raw(triple, entityTotal):
	newTriple = deepcopy(triple)
	oldHead = triple.h
	while True:
		newHead = random.randrange(entityTotal)
		if newHead != oldHead:
			break
	newTriple.h = newHead
	return newTriple

# Change the tail of a triple randomly,
# without checking whether it is a false negative sample.
def corrupt_tail_raw(triple, entityTotal):
	newTriple = deepcopy(triple)
	oldTail = triple.t
	while True:
		newTail = random.randrange(entityTotal)
		if newTail != oldTail:
			break
	newTriple.t = newTail
	return newTriple

# Change the head of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_head_filter(triple, entityTotal, tripleDict):
	newTriple = deepcopy(triple)
	while True:
		newHead = random.randrange(entityTotal)
		if (newHead, newTriple.t, newTriple.r) not in tripleDict:
			break
	newTriple.h = newHead
	return newTriple

# Change the tail of a triple randomly,
# with checking whether it is a false negative sample.
# If it is, regenerate.
def corrupt_tail_filter(triple, entityTotal, tripleDict):
	newTriple = deepcopy(triple)
	while True:
		newTail = random.randrange(entityTotal)
		if (newTriple.h, newTail, newTriple.r) not in tripleDict:
			break
	newTriple.t = newTail
	return newTriple

# Split the tripleList into #num_batches batches
def getBatchList(tripleList, num_batches):
	batchSize = len(tripleList) // num_batches
	batchList = [0] * num_batches
	for i in range(num_batches - 1):
		batchList[i] = tripleList[i * batchSize : (i + 1) * batchSize]
	batchList[num_batches - 1] = tripleList[(num_batches - 1) * batchSize : ]
	return batchList

def getThreeElements(tripleList):
	headList = [triple.h for triple in tripleList]
	tailList = [triple.t for triple in tripleList]
	relList = [triple.r for triple in tripleList]
	return headList, tailList, relList

# Sample a batch of #batchSize triples from tripleList
def getBatch_clean_random(tripleList, batchSize):
	newTripleList = random.sample(tripleList, batchSize)
	ph, pt ,pr = getThreeElements(newTripleList)
	return ph, pt, pr

def getBatch_clean_all(tripleList):
	ph, pt ,pr = getThreeElements(tripleList)
	return ph, pt, pr

# Corrupt the head or tail according to Bernoulli Distribution,
# (See "Knowledge Graph Embedding by Translating on Hyperplanes" paper)
# without checking whether it is a false negative sample.
def corrupt_raw_two_v2(triple, entityTotal, tail_per_head, head_per_tail):
	rel = triple.r
	split = tail_per_head[rel] / (tail_per_head[rel] + head_per_tail[rel])
	random_number = random.random()
	if random_number < split:
		newTriple = corrupt_head_raw(triple, entityTotal)
	else:
		newTriple = corrupt_tail_raw(triple, entityTotal)
	return newTriple

# Corrupt the head or tail according to Bernoulli Distribution,
# with checking whether it is a false negative sample.
def corrupt_filter_two_v2(triple, entityTotal, tripleDict, tail_per_head, head_per_tail):
	rel = triple.r
	split = tail_per_head[rel] / (tail_per_head[rel] + head_per_tail[rel])
	random_number = random.random()
	if random_number < split:
		newTriple = corrupt_head_filter(triple, entityTotal, tripleDict)
	else:
		newTriple = corrupt_tail_filter(triple, entityTotal, tripleDict)
	return newTriple

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_random(tripleList, batchSize, entityTotal):
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_head_raw(triple, entityTotal) if random.random() < 0.5 
		else corrupt_tail_raw(triple, entityTotal) for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# without checking whether false negative samples exist.
def getBatch_raw_all(tripleList, entityTotal):
	newTripleList = [corrupt_head_raw(triple, entityTotal) if random.random() < 0.5 
		else corrupt_tail_raw(triple, entityTotal) for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_random(tripleList, batchSize, entityTotal, tripleDict):
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5 
		else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail with equal probabilities,
# with checking whether false negative samples exist.
def getBatch_filter_all(tripleList, entityTotal, tripleDict):
	newTripleList = [corrupt_head_filter(triple, entityTotal, tripleDict) if random.random() < 0.5 
		else corrupt_tail_filter(triple, entityTotal, tripleDict) for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail according to Bernoulli Distribution,
# without checking whether false negative samples exist.
def getBatch_raw_random_v2(tripleList, batchSize, entityTotal, tail_per_head, head_per_tail):
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_raw_two_v2(triple, entityTotal, tail_per_head, head_per_tail) 
		for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail according to Bernoulli Distribution,
# without checking whether false negative samples exist.
def getBatch_raw_all_v2(tripleList, entityTotal, tail_per_head, head_per_tail):
	newTripleList = [corrupt_raw_two_v2(triple, entityTotal, tail_per_head, head_per_tail) 
		for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Sample a batch of #batchSize triples from tripleList,
# and generate negative samples by corrupting head or tail according to Bernoulli Distribution,
# with checking whether false negative samples exist.
def getBatch_filter_random_v2(tripleList, batchSize, entityTotal, tripleDict, tail_per_head, head_per_tail):
	oldTripleList = random.sample(tripleList, batchSize)
	newTripleList = [corrupt_filter_two_v2(triple, entityTotal, tripleDict, tail_per_head, head_per_tail) 
		for triple in oldTripleList]
	ph, pt ,pr = getThreeElements(oldTripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr

# Use all the tripleList,
# and generate negative samples by corrupting head or tail according to Bernoulli Distribution,
# with checking whether false negative samples exist.
def getBatch_filter_all_v2(tripleList, entityTotal, tripleDict, tail_per_head, head_per_tail):
	newTripleList = [corrupt_filter_two_v2(triple, entityTotal, tripleDict, tail_per_head, head_per_tail) 
		for triple in tripleList]
	ph, pt ,pr = getThreeElements(tripleList)
	nh, nt, nr = getThreeElements(newTripleList)
	return ph, pt, pr, nh, nt, nr
