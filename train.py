import os

class Config:
    name = 'fp_exp3'
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
    base_dir = '/content/drive/MyDrive/feedback_prize'
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