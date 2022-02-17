import os
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import gc
from collections import defaultdict
# nlp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast, AutoConfig, AutoModel, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

class Config:
    name = 'fp_exp1'
    # choose model 
    #model_savename = 'roberta-base'
    #model_savename = 'roberta-large'
    model_savename = 'longformer'
    # customize for my own Google Colab Environment
    if model_savename == 'longformer':
        model_name = 'allenai/longformer-base-4096'
    elif model_savename == 'roberta-base':
        model_name = 'roberta-base'
    elif model_savename == 'roberta-large':
        model_name = 'roberta-large'
    base_dir = '/root/kaggle/feedback-prize-2021'
    data_dir = os.path.join(base_dir, 'data')
    pre_data_dir = os.path.join(base_dir, 'data/preprocessed')
    model_dir = os.path.join(base_dir, f'model/{name}')
    output_dir = os.path.join(base_dir, f'output/{name}')
    is_debug = False
    n_epoch = 2 # not to exceed runtime limit
    n_fold = 5
    verbose_steps = 500
    random_seed = 42

    if model_savename == 'longformer':
        max_length = 1024
        inference_max_length = 4096
        train_batch_size = 4
        valid_batch_size = 4
        lr = 4e-5
    elif model_savename == 'roberta-base':
        max_length = 512
        inference_max_length = 512
        train_batch_size = 8
        valid_batch_size = 8
        lr = 8e-5
    elif model_savename == 'roberta-large':
        max_length = 512
        inference_max_length = 512
        train_batch_size = 4
        valid_batch_size = 4
        lr = 1e-5
    num_labels = 15
    label_subtokens = True
    output_hidden_states = True
    hidden_dropout_prob = 0.1
    layer_norm_eps = 1e-7
    add_pooling_layer = False
    verbose_steps = 500
    if is_debug:
        debug_sample = 1000
        verbose_steps = 16
        n_epoch = 1
        n_fold = 2

IGNORE_INDEX = -100
NON_LABEL = -1
OUTPUT_LABELS = ['0', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
LABELS_TO_IDS = {v:k for k,v in enumerate(OUTPUT_LABELS)}
IDS_TO_LABELS = {k:v for k,v in enumerate(OUTPUT_LABELS)}

MIN_THRESH = {
    "I-Lead": 9,
    "I-Position": 5,
    "I-Evidence": 14,
    "I-Claim": 3,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

PROB_THRESH = {
    "I-Lead": 0.7,
    "I-Position": 0.55,
    "I-Evidence": 0.65,
    "I-Claim": 0.55,
    "I-Concluding Statement": 0.7,
    "I-Counterclaim": 0.5,
    "I-Rebuttal": 0.55,
}

if not os.path.exists(Config.model_dir):
    os.makedirs(Config.model_dir)
if not os.path.exists(Config.output_dir):
    os.makedirs(Config.output_dir)