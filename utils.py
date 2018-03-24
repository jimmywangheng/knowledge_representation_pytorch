#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-11-13 22:09:37
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
from copy import deepcopy
import pickle
import random
import numpy as np
import time
import datetime

import loss

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

# Compare two Triples in the order of head, relation and tail
def cmp_head(a, b):
	return (a.h < b.h or (a.h == b.h and a.r < b.r) or (a.h == b.h and a.r == b.r and a.t < b.t))

# Compare two Triples in the order of tail, relation and head
def cmp_tail(a, b):
	return (a.t < b.t or (a.t == b.t and a.r < b.r) or (a.t == b.t and a.r == b.r and a.h < b.h))

# Compare two Triples in the order of relation, head and tail
def cmp_rel(a, b):
	return (a.r < b.r or (a.r == b.r and a.h < b.h) or (a.r == b.r and a.h == b.h and a.t < b.t))

def minimal(a, b):
	if a > b:
		return b
	return a

def cmp_list(a, b):
	return (minimal(a.h, a.t) > minimal(b.h, b.t))

emptyTriple = Triple(0, 0, 0)

# Calculate the statistics of datasets
def calculate(datasetPath):
	with open(os.path.join(datasetPath, 'relation2id.txt'), 'r') as fr:
		for line in fr:
			relationTotal = int(line)
			break

	freqRel = [0] * relationTotal # The frequency of each relation

	with open(os.path.join(datasetPath, 'entity2id.txt'), 'r') as fr:
		for line in fr:
			entityTotal = int(line)
			break

	freqEnt = [0] * entityTotal # The frequency of each entity

	tripleHead = []
	tripleTail = []
	tripleList = []

	tripleTotal = 0
	with open(os.path.join(datasetPath, 'train2id.txt'), 'r') as fr:
		i = 0
		for line in fr:
			# Ignore the first line, which is the number of triples
			if i == 0:
				i += 1
				continue
			else:
				line_split = line.split()
				head = int(line_split[0])
				tail = int(line_split[1])
				rel = int(line_split[2])
				tripleHead.append(Triple(head, tail, rel))
				tripleTail.append(Triple(head, tail, rel))
				tripleList.append(Triple(head, tail, rel))
				freqEnt[head] += 1
				freqEnt[tail] += 1
				freqRel[rel] += 1
				tripleTotal += 1

	with open(os.path.join(datasetPath, 'valid2id.txt'), 'r') as fr:
		i = 0
		for line in fr:
			if i == 0:
				i += 1
				continue
			else:
				line_split = line.split()
				head = int(line_split[0])
				tail = int(line_split[1])
				rel = int(line_split[2])
				tripleHead.append(Triple(head, tail, rel))
				tripleTail.append(Triple(head, tail, rel))
				tripleList.append(Triple(head, tail, rel))
				freqEnt[head] += 1
				freqEnt[tail] += 1
				freqRel[rel] += 1
				tripleTotal += 1

	with open(os.path.join(datasetPath, 'test2id.txt'), 'r') as fr:
		i = 0
		for line in fr:
			if i == 0:
				i += 1
				continue
			else:
				line_split = line.split()
				head = int(line_split[0])
				tail = int(line_split[1])
				rel = int(line_split[2])
				tripleHead.append(Triple(head, tail, rel))
				tripleTail.append(Triple(head, tail, rel))
				tripleList.append(Triple(head, tail, rel))
				freqEnt[head] += 1
				freqEnt[tail] += 1
				freqRel[rel] += 1
				tripleTotal += 1

	tripleHead.sort(key=lambda x: (x.h, x.r, x.t))
	tripleTail.sort(key=lambda x: (x.t, x.r, x.h))

	headDict = {}
	tailDict = {}
	for triple in tripleList:
		if triple.r not in headDict:
			headDict[triple.r] = {}
			tailDict[triple.r] = {}
			headDict[triple.r][triple.h] = set([triple.t])
			tailDict[triple.r][triple.t] = set([triple.h])
		else:
			if triple.h not in headDict[triple.r]:
				headDict[triple.r][triple.h] = set([triple.t])
			else:
				headDict[triple.r][triple.h].add(triple.t)

			if triple.t not in tailDict[triple.r]:
				tailDict[triple.r][triple.t] = set([triple.h])
			else:
				tailDict[triple.r][triple.t].add(triple.h)

	tail_per_head = [0] * relationTotal
	head_per_tail = [0] * relationTotal

	for rel in headDict:
		heads = headDict[rel].keys()
		tails = headDict[rel].values()
		totalHeads = len(heads)
		totalTails = sum([len(elem) for elem in tails])
		tail_per_head[rel] = totalTails / totalHeads

	for rel in tailDict:
		tails = tailDict[rel].keys()
		heads = tailDict[rel].values()
		totalTails = len(tails)
		totalHeads = sum([len(elem) for elem in heads])
		head_per_tail[rel] = totalHeads / totalTails

	connectedTailDict = {}
	for rel in headDict:
		if rel not in connectedTailDict:
			connectedTailDict[rel] = set()
		for head in headDict[rel]:
			connectedTailDict[rel] = connectedTailDict[rel].union(headDict[rel][head])

	connectedHeadDict = {}
	for rel in tailDict:
		if rel not in connectedHeadDict:
			connectedHeadDict[rel] = set()
		for tail in tailDict[rel]:
			connectedHeadDict[rel] = connectedHeadDict[rel].union(tailDict[rel][tail])

	print(tail_per_head)
	print(head_per_tail)

	listTripleHead = [(triple.h, triple.t, triple.r) for triple in tripleHead]
	listTripleTail = [(triple.h, triple.t, triple.r) for triple in tripleTail]
	listTripleList = [(triple.h, triple.t, triple.r) for triple in tripleList]
	with open(os.path.join(datasetPath, 'head_tail_proportion.pkl'), 'wb') as fw:
		pickle.dump(tail_per_head, fw)
		pickle.dump(head_per_tail, fw)

	with open(os.path.join(datasetPath, 'head_tail_connection.pkl'), 'wb') as fw:
		pickle.dump(connectedTailDict, fw)
		pickle.dump(connectedHeadDict, fw)

def getRel(triple):
    return triple.r

def getAnythingTotal(inPath, fileName):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		for line in fr:
			return int(line)

def loadTriple(inPath, fileName):
	with open(os.path.join(inPath, fileName), 'r') as fr:
		i = 0
		tripleList = []
		for line in fr:
			if i == 0:
				tripleTotal = int(line)
				i += 1
			else:
				line_split = line.split()
				head = int(line_split[0])
				tail = int(line_split[1])
				rel = int(line_split[2])
				tripleList.append(Triple(head, tail, rel))

	tripleDict = {}
	for triple in tripleList:
		tripleDict[(triple.h, triple.t, triple.r)] = True

	return tripleTotal, tripleList, tripleDict

def which_loss_type(num):
	if num == 0:
		return loss.marginLoss
	elif num == 1:
		return loss.EMLoss
	elif num == 2:
		return loss.WGANLoss
	elif num == 3:
		return nn.MSELoss