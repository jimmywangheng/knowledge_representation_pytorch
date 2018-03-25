#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2018-01-02 01:02:57
# @Author  : jimmy (jimmywangheng@qq.com)
# @Link    : http://sdcs.sysu.edu.cn
# @Version : $Id$

import os

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import time
import datetime
import random

from utils import *
from data import *
from evaluation import *
import loss
import model

from hyperboard import Agent

USE_CUDA = torch.cuda.is_available()

"""
The meaning of parameters:
self.dataset: Which dataset is used to train the model? Such as 'FB15k', 'WN18', etc.
self.learning_rate1: Learning rate (lr) for the first phase, when the evaluation result is dropping significantly.
self.learning_rate2: Initial lr for the second phase, when the evaluation result is slowly improving.
self.early_stopping_round: How many times will lr decrease? If set to 0, it remains constant.
self.L1_flag: If set to True, use L1 distance as dissimilarity; else, use L2.
self.ent_embedding_size: The embedding size of entities.
self.rel_embedding_size: The embedding size of relations.
self.num_batches: How many batches to train in one epoch?
self.train_times: The maximum number of epochs for training.
self.margin: The margin set for MarginLoss.
self.filter: Whether to check a generated negative sample is false negative.
self.momentum: The momentum of the optimizer.
self.optimizer: Which optimizer to use? Such as SGD, Adam, etc.
self.loss_function: Which loss function to use? Typically, we use margin loss.
self.entity_total: The number of different entities.
self.relation_total: The number of different relations.
self.batch_size: How many instances is contained in one batch?
"""

class Config(object):
	def __init__(self):
		self.dataset = None
		self.learning_rate1 = 0.001
		self.learning_rate2 = 0.0005
		self.early_stopping_round = 0
		self.L1_flag = True
		self.ent_embedding_size = 100
		self.rel_embedding_size = 100
		self.num_batches = 100
		self.train_times = 1000
		self.margin = 1.0
		self.filter = True
		self.momentum = 0.9
		self.optimizer = optim.Adam
		self.loss_function = loss.marginLoss
		self.entity_total = 0
		self.relation_total = 0
		self.batch_size = 0

if __name__ == "__main__":

	import argparse
	argparser = argparse.ArgumentParser()

	"""
	The meaning of some parameters:
	seed: Fix the random seed. Except for 0, which means no setting of random seed.
	port: The port number used by hyperboard, 
	which is a demo showing training curves in real time.
	You can refer to https://github.com/WarBean/hyperboard to know more.
	num_processes: Number of processes used to evaluate the result.	

	Note:
	Since we initialize the embeddings of entities and relations
	with the result of TransE,
	they have to be of the same size.
	In fact, embeddings of entities and relations can be of different sizes in TransR.
	"""

	argparser.add_argument('-d', '--dataset', type=str)
	argparser.add_argument('-l1', '--learning_rate1', type=float, default=0.001)
	argparser.add_argument('-l2', '--learning_rate2', type=float, default=0.0005)
	argparser.add_argument('-es', '--early_stopping_round', type=int, default=0)
	argparser.add_argument('-L', '--L1_flag', type=int, default=1)
	argparser.add_argument('-em', '--embedding_size', type=int, default=100)
	argparser.add_argument('-nb', '--num_batches', type=int, default=100)
	argparser.add_argument('-n', '--train_times', type=int, default=1000)
	argparser.add_argument('-m', '--margin', type=float, default=1.0)
	argparser.add_argument('-f', '--filter', type=int, default=1)
	argparser.add_argument('-mo', '--momentum', type=float, default=0.9)
	argparser.add_argument('-s', '--seed', type=int, default=0)
	argparser.add_argument('-op', '--optimizer', type=int, default=1)
	argparser.add_argument('-lo', '--loss_type', type=int, default=0)
	argparser.add_argument('-p', '--port', type=int, default=5000)
	argparser.add_argument('-np', '--num_processes', type=int, default=4)

	args = argparser.parse_args()

	# Start the hyperboard agent
	agent = Agent(address='127.0.0.1', port=args.port)

	if args.seed != 0:
		torch.manual_seed(args.seed)

	trainTotal, trainList, trainDict = loadTriple('./data/' + args.dataset, 'train2id.txt')
	validTotal, validList, validDict = loadTriple('./data/' + args.dataset, 'valid2id.txt')
	tripleTotal, tripleList, tripleDict = loadTriple('./data/' + args.dataset, 'triple2id.txt')
	config = Config()
	config.dataset = args.dataset
	config.learning_rate = args.learning_rate1

	config.early_stopping_round = args.early_stopping_round

	if args.L1_flag == 1:
		config.L1_flag = True
	else:
		config.L1_flag = False

	config.ent_embedding_size = args.embedding_size
	config.rel_embedding_size = args.embedding_size
	config.num_batches = args.num_batches
	config.train_times = args.train_times
	config.margin = args.margin

	if args.filter == 1:
		config.filter = True
	else:
		config.filter = False

	config.momentum = args.momentum

	if args.optimizer == 0:
		config.optimizer = optim.SGD
	elif args.optimizer == 1:
		config.optimizer = optim.Adam
	elif args.optimizer == 2:
		config.optimizer = optim.RMSprop

	if args.loss_type == 0:
		config.loss_function = loss.marginLoss

	config.entity_total = getAnythingTotal('./data/' + config.dataset, 'entity2id.txt')
	config.relation_total = getAnythingTotal('./data/' + config.dataset, 'relation2id.txt')
	config.batch_size = trainTotal // config.num_batches

	shareHyperparameters = {'dataset': args.dataset,
		'learning_rate1': args.learning_rate1,
		'learning_rate2': args.learning_rate2,
		'early_stopping_round': args.early_stopping_round,
		'L1_flag': args.L1_flag,
		'ent_embedding_size': args.embedding_size,
		'rel_embedding_size': args.embedding_size,
		'margin': args.margin,
		'filter': args.filter,
		'momentum': args.momentum,
		'seed': args.seed,
		'optimizer': args.optimizer,
		'loss_type': args.loss_type,
		}

	trainHyperparameters = shareHyperparameters.copy()
	trainHyperparameters.update({'type': 'train_loss'})

	validHyperparameters = shareHyperparameters.copy()
	validHyperparameters.update({'type': 'valid_loss'})

	hit10Hyperparameters = shareHyperparameters.copy()
	hit10Hyperparameters.update({'type': 'hit10'})

	meanrankHyperparameters = shareHyperparameters.copy()
	meanrankHyperparameters.update({'type': 'mean_rank'})

	trainCurve = agent.register(trainHyperparameters, 'train loss', overwrite=True)
	validCurve = agent.register(validHyperparameters, 'valid loss', overwrite=True)
	hit10Curve = agent.register(hit10Hyperparameters, 'hit@10', overwrite=True)
	meanrankCurve = agent.register(meanrankHyperparameters, 'mean rank', overwrite=True)

	loss_function = config.loss_function()
	model = model.TransRPretrainModel(config)

	if USE_CUDA:
		model.cuda()
		loss_function.cuda()
		longTensor = torch.cuda.LongTensor
		floatTensor = torch.cuda.FloatTensor

	else:
		longTensor = torch.LongTensor
		floatTensor = torch.FloatTensor

	optimizer = config.optimizer(model.parameters(), lr=config.learning_rate)
	margin = autograd.Variable(floatTensor([config.margin]))

	start_time = time.time()

	filename = '_'.join(
		['l1', str(args.learning_rate1),
		 'l2', str(args.learning_rate2),
		 'es', str(args.early_stopping_round),
		 'L', str(args.L1_flag),
		 'eem', str(args.embedding_size),
		 'rem', str(args.embedding_size),
		 'nb', str(args.num_batches),
		 'n', str(args.train_times),
		 'm', str(args.margin),
		 'f', str(args.filter),
		 'mo', str(args.momentum),
		 's', str(args.seed),
		 'op', str(args.optimizer),
		 'lo', str(args.loss_type),]) + '_TransR.ckpt'

	trainBatchList = getBatchList(trainList, config.num_batches)

	phase = 0

	for epoch in range(config.train_times):
		total_loss = floatTensor([0.0])
		random.shuffle(trainBatchList)
		for batchList in trainBatchList:
			if config.filter == True:
				pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_all(batchList, 
					config.entity_total, tripleDict)
			else:
				pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_all(batchList, 
					config.entity_total)

			batch_entity_set = set(pos_h_batch + pos_t_batch + neg_h_batch + neg_t_batch)
			batch_relation_set = set(pos_r_batch + neg_r_batch)
			batch_entity_list = list(batch_entity_set)
			batch_relation_list = list(batch_relation_set)

			pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
			pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
			pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
			neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
			neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
			neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

			model.zero_grad()
			pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = model(pos_h_batch, 
				pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch)

			if args.loss_type == 0:
				losses = loss_function(pos, neg, margin)
			else:
				losses = loss_function(pos, neg)
			ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
			rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(pos_h_e) + loss.normLoss(pos_t_e) + loss.normLoss(neg_h_e) + loss.normLoss(neg_t_e)
			
			losses.backward()
			optimizer.step()
			total_loss += losses.data

		agent.append(trainCurve, epoch, total_loss[0])

		if epoch % 10 == 0:
			now_time = time.time()
			print(now_time - start_time)
			print("Train total loss: %d %f" % (epoch, total_loss[0]))

		if epoch % 10 == 0:
			if config.filter == True:
				pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_filter_random(validList, 
					config.batch_size, config.entity_total, tripleDict)
			else:
				pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch = getBatch_raw_random(validList, 
					config.batch_size, config.entity_total)
			pos_h_batch = autograd.Variable(longTensor(pos_h_batch))
			pos_t_batch = autograd.Variable(longTensor(pos_t_batch))
			pos_r_batch = autograd.Variable(longTensor(pos_r_batch))
			neg_h_batch = autograd.Variable(longTensor(neg_h_batch))
			neg_t_batch = autograd.Variable(longTensor(neg_t_batch))
			neg_r_batch = autograd.Variable(longTensor(neg_r_batch))

			pos, neg, pos_h_e, pos_t_e, neg_h_e, neg_t_e = model(pos_h_batch, 
				pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch)

			if args.loss_type == 0:
				losses = loss_function(pos, neg, margin)
			else:
				losses = loss_function(pos, neg)
			ent_embeddings = model.ent_embeddings(torch.cat([pos_h_batch, pos_t_batch, neg_h_batch, neg_t_batch]))
			rel_embeddings = model.rel_embeddings(torch.cat([pos_r_batch, neg_r_batch]))
			losses = losses + loss.normLoss(ent_embeddings) + loss.normLoss(rel_embeddings) + loss.normLoss(pos_h_e) + loss.normLoss(pos_t_e) + loss.normLoss(neg_h_e) + loss.normLoss(neg_t_e)

			print("Valid batch loss: %d %f" % (epoch, losses.data[0]))
			agent.append(validCurve, epoch, losses.data[0])

		if config.early_stopping_round > 0:
			if epoch == 0:
				ent_embeddings = model.ent_embeddings.weight.data
				rel_embeddings = model.rel_embeddings.weight.data
				proj_embeddings = model.proj_embeddings.weight.data
				L1_flag = model.L1_flag
				filter = model.filter	
				hit10, meanrank = evaluation_transR(validList, tripleDict, ent_embeddings, rel_embeddings, 
					proj_embeddings, L1_flag, filter, k=config.batch_size, num_processes=args.num_processes)
				worst_hit10 = hit10
				best_meanrank = meanrank
				agent.append(hit10Curve, epoch, hit10)
				agent.append(meanrankCurve, epoch, meanrank)
				torch.save(model, os.path.join('./model/' + args.dataset, filename))	
				#if USE_CUDA:
					#model.cuda()

			# Evaluate on validation set for every 5 epochs
			elif epoch % 5 == 0:
				ent_embeddings = model.ent_embeddings.weight.data
				rel_embeddings = model.rel_embeddings.weight.data
				proj_embeddings = model.proj_embeddings.weight.data
				L1_flag = model.L1_flag
				filter = model.filter
				hit10, meanrank = evaluation_transR(validList, tripleDict, ent_embeddings, rel_embeddings, 
					proj_embeddings, L1_flag, filter, k=config.batch_size, num_processes=args.num_processes)
				agent.append(hit10Curve, epoch, hit10)
				agent.append(meanrankCurve, epoch, meanrank)
				if phase == 0:
					if hit10 < worst_hit10:
						worst_hit10 = hit10
						best_meanrank = meanrank

					# When the evaluation result on validation set stops dropping,
					# the first phase terminates,
					# set lr to args.learning_rate2,
					# and the second phase begins.
					else:
						phase += 1
						optimizer.param_groups[0]['lr'] = args.learning_rate2
						best_epoch = epoch
						meanrank_not_decrease_time = 0
						lr_decrease_time = 0
				else:
					if meanrank < best_meanrank:
						meanrank_not_decrease_time = 0
						best_meanrank = meanrank
						torch.save(model, os.path.join('./model/' + args.dataset, filename))
					else:
						meanrank_not_decrease_time += 1

						# If the result hasn't improved for consecutive 5 evaluations, decrease learning rate
						if meanrank_not_decrease_time == 5:
							lr_decrease_time += 1
							if lr_decrease_time == config.early_stopping_round:
								break
							else:
								optimizer.param_groups[0]['lr'] *= 0.5
								meanrank_not_decrease_time = 0
				#if USE_CUDA:
					#model.cuda()

		elif (epoch + 1) % 10 == 0 or epoch == 0:
			torch.save(model, os.path.join('./model/' + args.dataset, filename))

	testTotal, testList, testDict = loadTriple('./data/' + args.dataset, 'test2id.txt')
	oneToOneTotal, oneToOneList, oneToOneDict = loadTriple('./data/' + args.dataset, 'one_to_one.txt')
	oneToManyTotal, oneToManyList, oneToManyDict = loadTriple('./data/' + args.dataset, 'one_to_many.txt')
	manyToOneTotal, manyToOneList, manyToOneDict = loadTriple('./data/' + args.dataset, 'many_to_one.txt')
	manyToManyTotal, manyToManyList, manyToManyDict = loadTriple('./data/' + args.dataset, 'many_to_many.txt')
	
	ent_embeddings = model.ent_embeddings.weight.data
	rel_embeddings = model.rel_embeddings.weight.data
	proj_embeddings = model.proj_embeddings.weight.data
	L1_flag = model.L1_flag
	filter = model.filter
	
	hit10Test, meanrankTest = evaluation_transR(testList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=0)

	hit10OneToOneHead, meanrankOneToOneHead = evaluation_transR(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=1)
	hit10OneToManyHead, meanrankOneToManyHead = evaluation_transR(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=1)
	hit10ManyToOneHead, meanrankManyToOneHead = evaluation_transR(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=1)
	hit10ManyToManyHead, meanrankManyToManyHead = evaluation_transR(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=1)

	hit10OneToOneTail, meanrankOneToOneTail = evaluation_transR(oneToOneList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=2)
	hit10OneToManyTail, meanrankOneToManyTail = evaluation_transR(oneToManyList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=2)
	hit10ManyToOneTail, meanrankManyToOneTail = evaluation_transR(manyToOneList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=2)
	hit10ManyToManyTail, meanrankManyToManyTail = evaluation_transR(manyToManyList, tripleDict, ent_embeddings, rel_embeddings, proj_embeddings, L1_flag, filter, head=2)	

	writeList = [filename, 
		'testSet', '%.6f' % hit10Test, '%.6f' % meanrankTest, 
		'one_to_one_head', '%.6f' % hit10OneToOneHead, '%.6f' % meanrankOneToOneHead, 
		'one_to_many_head', '%.6f' % hit10OneToManyHead, '%.6f' % meanrankOneToManyHead, 
		'many_to_one_head', '%.6f' % hit10ManyToOneHead, '%.6f' % meanrankManyToOneHead, 
		'many_to_many_head', '%.6f' % hit10ManyToManyHead, '%.6f' % meanrankManyToManyHead,
		'one_to_one_tail', '%.6f' % hit10OneToOneTail, '%.6f' % meanrankOneToOneTail, 
		'one_to_many_tail', '%.6f' % hit10OneToManyTail, '%.6f' % meanrankOneToManyTail, 
		'many_to_one_tail', '%.6f' % hit10ManyToOneTail, '%.6f' % meanrankManyToOneTail, 
		'many_to_many_tail', '%.6f' % hit10ManyToManyTail, '%.6f' % meanrankManyToManyTail,]

	# Write the result into file
	with open(os.path.join('./result/', args.dataset + '.txt'), 'a') as fw:
		fw.write('\t'.join(writeList) + '\n')

