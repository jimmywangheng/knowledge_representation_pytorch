#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2017-12-15 16:03:42
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os
import numpy as np
import time
import datetime
import random
import multiprocessing
import math

from itertools import groupby

from utils import Triple, getRel

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics.pairwise import pairwise_distances

from projection import *

USE_CUDA = torch.cuda.is_available()

if USE_CUDA:
	longTensor = torch.cuda.LongTensor
	floatTensor = torch.cuda.FloatTensor

else:
	longTensor = torch.LongTensor
	floatTensor = torch.FloatTensor

def isHit10(triple, tree, cal_embedding, tripleDict, isTail):
	# If isTail == True, evaluate the prediction of tail entity
	if isTail == True:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			tail_dist, tail_ind = tree.query(cal_embedding, k=k)
			for elem in tail_ind[0][k - 15: k]:
				if triple.t == elem:
					return True
				elif (triple.h, elem, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False
	# If isTail == False, evaluate the prediction of head entity
	else:
		k = 0
		wrongCount = 0
		while wrongCount < 10:
			k += 15
			head_dist, head_ind = tree.query(cal_embedding, k=k)
			for elem in head_ind[0][k - 15: k]:
				if triple.h == elem:
					return True
				elif (elem, triple.t, triple.r) in tripleDict:
					continue
				else:
					wrongCount += 1
					if wrongCount > 9:
						return False

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == tail:
			return wrongAnswer
		elif (head, num, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

# Find the rank of ground truth head in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereHead(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == head:
			return wrongAnswer
		elif (num, tail, rel) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

def pairwise_L1_distances(A, B):
	dist = torch.sum(torch.abs(A.unsqueeze(1) - B.unsqueeze(0)), dim=2)
	return dist

def pairwise_L2_distances(A, B):
	AA = torch.sum(A ** 2, dim=1).unsqueeze(1)
	BB = torch.sum(B ** 2, dim=1).unsqueeze(0)
	dist = torch.mm(A, torch.transpose(B, 0, 1))
	dist *= -2
	dist += AA
	dist += BB
	return dist

def evaluation_transE_helper(testList, tripleDict, ent_embeddings, 
	rel_embeddings, L1_flag, filter, head=0):
	# embeddings are numpy like

	headList = [triple.h for triple in testList]
	tailList = [triple.t for triple in testList]
	relList = [triple.r for triple in testList]

	h_e = ent_embeddings[headList]
	t_e = ent_embeddings[tailList]
	r_e = rel_embeddings[relList]

	# Evaluate the prediction of only head entities
	if head == 1:
		c_h_e = t_e - r_e

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		# Don't check whether it is false negative
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		# Check whether it is false negative
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListHead)
		hit10Count = len(isHit10ListHead)
		tripleCount = len(rankListHead)

	# Evaluate the prediction of only tail entities
	elif head == 2:
		c_t_e = h_e + r_e

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		totalRank = sum(rankListTail)
		hit10Count = len(isHit10ListTail)
		tripleCount = len(rankListTail)

	# Evaluate the prediction of both head and tail entities
	else:
		c_t_e = h_e + r_e
		c_h_e = t_e - r_e

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, ent_embeddings, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, ent_embeddings, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, ent_embeddings, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, ent_embeddings, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListTail) + sum(rankListHead)
		hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
		tripleCount = len(rankListTail) + len(rankListHead)

	return hit10Count, totalRank, tripleCount

class MyProcessTransE(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, L1_flag, filter, queue=None, head=0):
		super(MyProcessTransE, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transE_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, L1_flag, filter, head)

		L.append((hit10Count, totalRank, tripleCount))

# Use multiprocessing to speed up evaluation
def evaluation_transE(testList, tripleDict, ent_embeddings, rel_embeddings, 
	L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
	# embeddings are numpy like

	if k > len(testList):
		testList = random.choices(testList, k=k)
	elif k > 0:
		testList = random.sample(testList, k=k)

	# Split the testList into #num_processes parts
	len_split = math.ceil(len(testList) / num_processes)
	testListSplit = [testList[i : i + len_split] for i in range(0, len(testList), len_split)]

	with multiprocessing.Manager() as manager:
		# Create a public writable list to store the result
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransE(L, tripleDict, ent_embeddings, rel_embeddings,
				L1_flag, filter, queue=queue, head=head)
			workerList.append(worker)
			worker.daemon = True
			worker.start()

		for subList in testListSplit:
			queue.put(subList)

		queue.join()

		resultList = list(L)

		# Terminate the worker after execution, to avoid memory leaking
		for worker in workerList:
			worker.terminate()

	if head == 1 or head == 2:
		hit10 = sum([elem[0] for elem in resultList]) / len(testList)
		meanrank = sum([elem[1] for elem in resultList]) / len(testList)
	else:
		hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
		meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

	print('Meanrank: %.6f' % meanrank)
	print('Hit@10: %.6f' % hit10)

	return hit10, meanrank		

def evaluation_transH_helper(testList, tripleDict, ent_embeddings, 
	rel_embeddings, norm_embeddings, L1_flag, filter, head=0):
	# embeddings are torch tensor like (No Variable!)
	# Only one kind of relation

	headList = [triple.h for triple in testList]
	tailList = [triple.t for triple in testList]
	relList = [triple.r for triple in testList]

	h_e = ent_embeddings[headList]
	t_e = ent_embeddings[tailList]
	r_e = rel_embeddings[relList]
	this_rel = relList[0]
	this_norm_emb = norm_embeddings[this_rel]
	this_proj_all_e = projection_transH_pytorch(ent_embeddings, this_norm_emb)
	this_proj_all_e = this_proj_all_e.cpu().numpy()

	if head == 1:
		proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
		c_h_e = proj_t_e - r_e
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListHead)
		hit10Count = len(isHit10ListHead)
		tripleCount = len(rankListHead)

	elif head == 2:
		proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
		c_t_e = proj_h_e + r_e
		c_t_e = c_t_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		totalRank = sum(rankListTail)
		hit10Count = len(isHit10ListTail)
		tripleCount = len(rankListTail)

	else:
		proj_h_e = projection_transH_pytorch(h_e, this_norm_emb)
		c_t_e = proj_h_e + r_e
		proj_t_e = projection_transH_pytorch(t_e, this_norm_emb)
		c_h_e = proj_t_e - r_e

		c_t_e = c_t_e.cpu().numpy()
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListTail) + sum(rankListHead)
		hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
		tripleCount = len(rankListTail) + len(rankListHead)

	return hit10Count, totalRank, tripleCount

class MyProcessTransH(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, norm_embeddings, L1_flag, filter, queue=None, head=0):
		super(MyProcessTransH, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.norm_embeddings = norm_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.norm_embeddings, self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		norm_embeddings, L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transH_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, norm_embeddings, L1_flag, filter, head)

		L.append((hit10Count, totalRank, tripleCount))

def evaluation_transH(testList, tripleDict, ent_embeddings, rel_embeddings, 
	norm_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
	# embeddings are torch tensor like (No Variable!)

	if k > len(testList):
		testList = random.choices(testList, k=k)
	elif k > 0:
		testList = random.sample(testList, k=k)

	# Split the testList according to the relation
	testList.sort(key=lambda x: (x.r, x.h, x.t))
	grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

	ent_embeddings = ent_embeddings.cpu()
	rel_embeddings = rel_embeddings.cpu()
	norm_embeddings = norm_embeddings.cpu()

	with multiprocessing.Manager() as manager:
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransH(L, tripleDict, ent_embeddings, rel_embeddings,
				norm_embeddings, L1_flag, filter, queue=queue, head=head)
			workerList.append(worker)
			worker.daemon = True
			worker.start()

		for k, subList in grouped:
			queue.put(subList)

		queue.join()

		resultList = list(L)

		for worker in workerList:
			worker.terminate()

	if head == 1 or head == 2:
		hit10 = sum([elem[0] for elem in resultList]) / len(testList)
		meanrank = sum([elem[1] for elem in resultList]) / len(testList)
	else:
		hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
		meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

	print('Meanrank: %.6f' % meanrank)
	print('Hit@10: %.6f' % hit10)

	return hit10, meanrank

def evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
	rel_embeddings, proj_embeddings, L1_flag, filter, head):
	# embeddings are torch tensor like (No Variable!)
	# Only one kind of relation

	headList = [triple.h for triple in testList]
	tailList = [triple.t for triple in testList]
	relList = [triple.r for triple in testList]

	h_e = ent_embeddings[headList]
	t_e = ent_embeddings[tailList]
	r_e = rel_embeddings[relList]
	this_rel = relList[0]
	this_proj_emb = proj_embeddings[[this_rel]]
	this_proj_all_e = projection_transR_pytorch(ent_embeddings, this_proj_emb)
	this_proj_all_e = this_proj_all_e.cpu().numpy()

	if head == 1:
		proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
		c_h_e = proj_t_e - r_e
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListHead)
		hit10Count = len(isHit10ListHead)
		tripleCount = len(rankListHead)

	elif head == 2:
		proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
		c_t_e = proj_h_e + r_e
		c_t_e = c_t_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		totalRank = sum(rankListTail)
		hit10Count = len(isHit10ListTail)
		tripleCount = len(rankListTail)

	else:
		proj_h_e = projection_transR_pytorch(h_e, this_proj_emb)
		c_t_e = proj_h_e + r_e
		proj_t_e = projection_transR_pytorch(t_e, this_proj_emb)
		c_h_e = proj_t_e - r_e

		c_t_e = c_t_e.cpu().numpy()
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListTail) + sum(rankListHead)
		hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
		tripleCount = len(rankListTail) + len(rankListHead)

	return hit10Count, totalRank, tripleCount

class MyProcessTransR(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, proj_embeddings, L1_flag, filter, queue=None, head=0):
		super(MyProcessTransR, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.proj_embeddings = proj_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		proj_embeddings, L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transR_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, proj_embeddings, L1_flag, filter, head=head)

		L.append((hit10Count, totalRank, tripleCount))

def evaluation_transR(testList, tripleDict, ent_embeddings, rel_embeddings, 
	proj_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
	# embeddings are torch tensor like (No Variable!)

	if k > len(testList):
		testList = random.choices(testList, k=k)
	elif k > 0:
		testList = random.sample(testList, k=k)

	# Split the testList according to the relation
	testList.sort(key=lambda x: (x.r, x.h, x.t))
	grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

	ent_embeddings = ent_embeddings.cpu()
	rel_embeddings = rel_embeddings.cpu()
	proj_embeddings = proj_embeddings.cpu()

	with multiprocessing.Manager() as manager:
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransR(L, tripleDict, ent_embeddings, rel_embeddings,
				proj_embeddings, L1_flag, filter, queue=queue, head=head)
			workerList.append(worker)
			worker.daemon = True
			worker.start()

		for k, subList in grouped:
			queue.put(subList)

		queue.join()

		resultList = list(L)

		for worker in workerList:
			worker.terminate()

	if head == 1 or head == 2:
		hit10 = sum([elem[0] for elem in resultList]) / len(testList)
		meanrank = sum([elem[1] for elem in resultList]) / len(testList)	
	else:
		hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
		meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

	print('Meanrank: %.6f' % meanrank)
	print('Hit@10: %.6f' % hit10)

	return hit10, meanrank

def evaluation_transD_helper(testList, tripleDict, ent_embeddings, 
	rel_embeddings, ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, head=0):
	# embeddings are torch tensor like (No Variable!)
	# Only one kind of relation

	headList = [triple.h for triple in testList]
	tailList = [triple.t for triple in testList]
	relList = [triple.r for triple in testList]

	h_e = ent_embeddings[headList]
	t_e = ent_embeddings[tailList]
	r_e = rel_embeddings[relList]
	head_proj_emb = ent_proj_embeddings[headList]
	tail_proj_emb = ent_proj_embeddings[tailList]
	this_rel = relList[0]
	this_rel_proj_emb = rel_proj_embeddings[[this_rel]]
	this_proj_all_e = projection_transD_pytorch_samesize(ent_embeddings, ent_proj_embeddings, this_rel_proj_emb)
	this_proj_all_e = this_proj_all_e.cpu().numpy()

	if head == 1:
		proj_t_e = projection_transD_pytorch_samesize(t_e, tail_proj_emb, this_rel_proj_emb)
		c_h_e = proj_t_e - r_e
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListHead)
		hit10Count = len(isHit10ListHead)
		tripleCount = len(rankListHead)

	elif head == 2:
		proj_h_e = projection_transD_pytorch_samesize(h_e, head_proj_emb, this_rel_proj_emb)
		c_t_e = proj_h_e + r_e
		c_t_e = c_t_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		totalRank = sum(rankListTail)
		hit10Count = len(isHit10ListTail)
		tripleCount = len(rankListTail)

	else:
		proj_h_e = projection_transD_pytorch_samesize(h_e, head_proj_emb, this_rel_proj_emb)
		c_t_e = proj_h_e + r_e
		proj_t_e = projection_transD_pytorch_samesize(t_e, tail_proj_emb, this_rel_proj_emb)
		c_h_e = proj_t_e - r_e

		c_t_e = c_t_e.cpu().numpy()
		c_h_e = c_h_e.cpu().numpy()

		if L1_flag == True:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_t_e, this_proj_all_e, metric='euclidean')

		rankArrayTail = np.argsort(dist, axis=1)
		if filter == False:
			rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
		else:
			rankListTail = [argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayTail)]

		isHit10ListTail = [x for x in rankListTail if x < 10]

		if L1_flag == True:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='manhattan')
		else:
			dist = pairwise_distances(c_h_e, this_proj_all_e, metric='euclidean')

		rankArrayHead = np.argsort(dist, axis=1)
		if filter == False:
			rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
		else:
			rankListHead = [argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict) 
							for elem in zip(headList, tailList, relList, rankArrayHead)]

		isHit10ListHead = [x for x in rankListHead if x < 10]

		totalRank = sum(rankListTail) + sum(rankListHead)
		hit10Count = len(isHit10ListTail) + len(isHit10ListHead)
		tripleCount = len(rankListTail) + len(rankListHead)

	return hit10Count, totalRank, tripleCount

class MyProcessTransD(multiprocessing.Process):
	def __init__(self, L, tripleDict, ent_embeddings, 
		rel_embeddings, ent_proj_embeddings, rel_proj_embeddings,L1_flag, filter, queue=None, head=0):
		super(MyProcessTransD, self).__init__()
		self.L = L
		self.queue = queue
		self.tripleDict = tripleDict
		self.ent_embeddings = ent_embeddings
		self.rel_embeddings = rel_embeddings
		self.ent_proj_embeddings = ent_proj_embeddings
		self.rel_proj_embeddings = rel_proj_embeddings
		self.L1_flag = L1_flag
		self.filter = filter
		self.head = head

	def run(self):
		while True:
			testList = self.queue.get()
			try:
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.ent_proj_embeddings, self.rel_proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			except:
				time.sleep(5)
				self.process_data(testList, self.tripleDict, self.ent_embeddings, self.rel_embeddings,
					self.ent_proj_embeddings, self.rel_proj_embeddings, self.L1_flag, self.filter, self.L, self.head)
			self.queue.task_done()

	def process_data(self, testList, tripleDict, ent_embeddings, rel_embeddings, 
		ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, L, head):

		hit10Count, totalRank, tripleCount = evaluation_transD_helper(testList, tripleDict, ent_embeddings, 
			rel_embeddings, ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, head=head)

		L.append((hit10Count, totalRank, tripleCount))

def evaluation_transD(testList, tripleDict, ent_embeddings, rel_embeddings, 
	ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, k=0, num_processes=multiprocessing.cpu_count(), head=0):
	# embeddings are torch tensor like (No Variable!)

	if k > len(testList):
		testList = random.choices(testList, k=k)
	elif k > 0:
		testList = random.sample(testList, k=k)

	# Split the testList according to the relation
	testList.sort(key=lambda x: (x.r, x.h, x.t))
	grouped = [(k, list(g)) for k, g in groupby(testList, key=getRel)]

	ent_embeddings = ent_embeddings.cpu()
	rel_embeddings = rel_embeddings.cpu()
	ent_proj_embeddings = ent_proj_embeddings.cpu()
	rel_proj_embeddings = rel_proj_embeddings.cpu()

	with multiprocessing.Manager() as manager:
		L = manager.list()
		queue = multiprocessing.JoinableQueue()
		workerList = []
		for i in range(num_processes):
			worker = MyProcessTransD(L, tripleDict, ent_embeddings, rel_embeddings,
				ent_proj_embeddings, rel_proj_embeddings, L1_flag, filter, queue=queue, head=head)
			workerList.append(worker)
			worker.daemon = True
			worker.start()

		for k, subList in grouped:
			queue.put(subList)

		queue.join()

		resultList = list(L)

		for worker in workerList:
			worker.terminate()

	if head == 1 or head == 2:
		hit10 = sum([elem[0] for elem in resultList]) / len(testList)
		meanrank = sum([elem[1] for elem in resultList]) / len(testList)
	else:
		hit10 = sum([elem[0] for elem in resultList]) / (2 * len(testList))
		meanrank = sum([elem[1] for elem in resultList]) / (2 * len(testList))

	print('Meanrank: %.6f' % meanrank)
	print('Hit@10: %.6f' % hit10)

	return hit10, meanrank
