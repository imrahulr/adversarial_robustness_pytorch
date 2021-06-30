"""
Adversarial Evaluation with PGD+, CW (Margin) PGD and black box adversary.
"""

import json
import time
import argparse
import shutil

import os
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm

import torch
import torch.nn as nn

from core.attacks import create_attack
from core.attacks import CWLoss

from core.data import get_data_info
from core.data import load_data

from core.models import create_model

from core.utils import ctx_noparamgrad_and_eval
from core.utils import Logger
from core.utils import parser_eval
from core.utils import seed
from core.utils import Trainer



# Setup

parse = parser_eval()
args = parse.parse_args()

LOG_DIR = args.log_dir + args.desc
with open(LOG_DIR+'/args.txt', 'r') as f:
    old = json.load(f)
    args.__dict__ = dict(vars(args), **old)

DATA_DIR = args.data_dir + args.data + '/'
LOG_DIR = args.log_dir + args.desc
WEIGHTS = LOG_DIR + '/weights-best.pt'

logger = Logger(LOG_DIR+'/log-adv.log')

info = get_data_info(DATA_DIR)
BATCH_SIZE = args.batch_size
BATCH_SIZE_VALIDATION = args.batch_size_validation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.log('Using device: {}'.format(device))



# Load data

seed(args.seed)
_, _, _, test_dataloader = load_data(DATA_DIR, BATCH_SIZE, BATCH_SIZE_VALIDATION, use_augmentation=False, 
                                     shuffle_train=False)



# Helper function

def eval_multiple_restarts(attack, model, dataloader, num_restarts=5, verbose=True):
    """
    Evaluate adversarial accuracy with multiple restarts.
    """
    model.eval()
    N = len(dataloader.dataset)
    is_correct = torch.ones(N).bool().to(device)
    for i in tqdm(range(0, num_restarts), disable=not verbose):
        iter_is_correct = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(model):
                x_adv, _ = attack.perturb(x, y)
            out = model(x_adv)
            iter_is_correct.extend(torch.softmax(out, dim=1).argmax(dim=1) == y)
        is_correct = torch.logical_and(is_correct, torch.BoolTensor(iter_is_correct).to(device))
    
    adv_acc = (is_correct.sum().float()/N).item()
    return adv_acc

def eval_multiple_restarts_advertorch(attack, model, dataloader, num_restarts=1, verbose=True):
    """
    Evaluate adversarial accuracy with multiple restarts (Advertorch).
    """
    model.eval()
    N = len(dataloader.dataset)
    is_correct = torch.ones(N).bool().to(device)
    for i in tqdm(range(0, num_restarts), disable=not verbose):
        iter_is_correct = []
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            with ctx_noparamgrad_and_eval(model):
                x_adv = attack.perturb(x, y)
            out = model(x_adv)
            iter_is_correct.extend(torch.softmax(out, dim=1).argmax(dim=1) == y)
        is_correct = torch.logical_and(is_correct, torch.BoolTensor(iter_is_correct).to(device))
    
    adv_acc = (is_correct.sum().float()/N).item()
    return adv_acc



# PGD Evaluation

seed(args.seed)
trainer = Trainer(info, args)
if 'tau' in args and args.tau:
    print ('Using WA model.')
trainer.load_model(WEIGHTS)
trainer.model.eval()

test_acc = trainer.eval(test_dataloader)
logger.log('\nStandard Accuracy-\tTest: {:.2f}%.'.format(test_acc*100))



if args.wb:    
    # CW-PGD-40 Evaluation
    seed(args.seed)
    num_restarts = 1
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0
    assert args.attack in ['linf-pgd', 'l2-pgd'], 'CW evaluation only supported for attack=linf-pgd or attack=l2-pgd !'
    attack = create_attack(trainer.model, CWLoss, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    logger.log('\n==== CW-PGD Evaluation. ====')
    logger.log('Attack: cw-{}.'.format(args.attack))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                args.attack_eps, 
                                                                                                args.attack_iter))

    test_adv_acc1 = eval_multiple_restarts(attack, trainer.model, test_dataloader, num_restarts,  verbose=False)
    logger.log('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc1*100))
    

    # PGD-40 (with 5 restarts) Evaluation
    seed(args.seed)
    num_restarts = 5
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0
    attack = create_attack(trainer.model, trainer.criterion, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    logger.log('\n==== PGD+ Evaluation. ====')
    logger.log('Attack: {} with {} restarts.'.format(args.attack, num_restarts))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                args.attack_eps, 
                                                                                                args.attack_iter))

    test_adv_acc2 = eval_multiple_restarts(attack, trainer.model, test_dataloader, num_restarts, verbose=True)
    logger.log('Adversarial Accuracy-\tTest: {:.2f}%.'.format(test_adv_acc2*100))



# Black Box Evaluation

class dotdict(dict):
    def __getattr__(self, name):
        return self[name]

if args.source != None:
    seed(args.seed)
    assert args.attack in ['linf-pgd', 'l2-pgd'], 'Black-box evaluation only supported for attack=linf-pgd or attack=l2-pgd!'
    if args.attack in ['linf-pgd', 'linf-df', 'fgsm']:        
        args.attack_iter, args.attack_step = 40, 0.01
    else:
        args.attack_iter, args.attack_step = 40, 30/255.0

    SRC_LOG_DIR = args.log_dir + args.source
    with open(SRC_LOG_DIR+'/args.txt', 'r') as f:
        src_args = json.load(f)
        src_args = dotdict(src_args)
    
    src_model = create_model(src_args.model, src_args.normalize, info, device)
    src_model.load_state_dict(torch.load(SRC_LOG_DIR + '/weights-best.pt')['model_state_dict'])
    src_model.eval()
    
    src_attack = create_attack(src_model, trainer.criterion, args.attack, args.attack_eps, args.attack_iter, args.attack_step)
    adv_acc = 0.0
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        with ctx_noparamgrad_and_eval(src_model):
            x_adv, _ = src_attack.perturb(x, y)            
        out = trainer.model(x_adv)
        adv_acc += accuracy(y, out)
    adv_acc /= len(test_dataloader)
    
    logger.log('\n==== Black-box Evaluation. ====')
    logger.log('Source Model: {}.'.format(args.source))
    logger.log('Attack: {}.'.format(args.attack))
    logger.log('Attack Parameters: Step size: {:.3f}, Epsilon: {:.3f}, #Iterations: {}.'.format(args.attack_step, 
                                                                                                   args.attack_eps, 
                                                                                                   args.attack_iter))
    logger.log('Black-box Adv. Accuracy-\tTest: {:.2f}%.'.format(adv_acc*100))
    del src_attack, src_model


logger.log('Script Completed.')