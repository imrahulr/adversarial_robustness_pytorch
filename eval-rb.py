"""
Evaluation with Robustbench.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from robustbench import benchmark

from core.data import get_data_info
from core.models import create_model

from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed



# Setup

parse = parser_eval()

args = parse.parse_args()

LOG_DIR = args.log_dir + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

args.data = 'cifar10' if args.data in ['cifar10s', 'cifar10g'] else args.data
DATA_DIR = args.data_dir + args.data + '/'
LOG_DIR = args.log_dir + args.desc
WEIGHTS = LOG_DIR + '/weights-best.pt'

log_path = LOG_DIR + f'/log-corr-{args.threat}.log'
logger = Logger(log_path)

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

assert args.data in ['cifar10'], 'Evaluation on Robustbench is only supported for cifar10!'

threat_model = args.threat
dataset = args.data
model_name = args.desc



# Model

model = create_model(args.model, args.normalize, info, device)
checkpoint = torch.load(WEIGHTS)
if 'tau' in args and args.tau:
    print ('Using WA model.')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
del checkpoint



# Common corruptions

seed(args.seed)
clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=args.num_samples, dataset=dataset,
                                  threat_model=threat_model, eps=args.attack_eps, device=device, to_disk=False, 
                                  data_dir=args.tmp_dir + args.data + 'c')


logger.log('Model: {}'.format(args.desc))
logger.log('Evaluating robustness on {} with threat model={}.'.format(args.data, args.threat))
logger.log('Clean Accuracy: \t{:.2f}%.\nRobust Accuracy: \t{:.2f}%.'.format(clean_acc*100, robust_acc*100))