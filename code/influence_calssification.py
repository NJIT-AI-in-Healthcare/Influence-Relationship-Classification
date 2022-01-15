# !/usr/bin/env python3
#  -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 18:59:42 2019

@author: LMD
"""
 
import pandas as pd 
from preprocessing import preprocessing
from model_util import model_util
from result_analysis import result_analysis
import random


# model options
'''
Baseline models:
    baseline_max: the baseline model used in the current paper
    baseline_avg: the baseline model used in the first manscript we submitted
    baseline_question_solo: the baseline model only using question module
    baseline_action_solo: the baseline model only using action module
ARC-I models:
    new_dot_arci: the influence model combining ARC-I and Q/A modules with dot architecture
    new_cat_arci: the influence model combining ARC-I and Q/A modules with cat architecture
Match Pyramid models:
    new_dot_match_pyramid: the influence model combining Match Pyramid and Q/A modules with dot architecture
    new_cat_match_pyramid: the influence model combining Match Pyramid and Q/A modules with cat architecture
'''
influence_model_name = 'baseline_max'# 'baseline_max'# 'new_dot_arci'# 'new_cat_match_pyramid'# 'simple_arci'# 'new_cat_arci'# 'new_dot_arci'
relevance_model_name = ''# 'new_match_pyramid'# 'new_match_pyramid'
if 'arci' in influence_model_name or 'baseline' in influence_model_name:
    relevance_model_name = 'new_arci'
elif 'match_pyramid' in influence_model_name:
    relevance_model_name = 'new_match_pyramid'

# data paths
relevance_folder_path = '../relevant_classification_model/'
influence_folder_path = '../influence_classification_model/'
relevance_model_path = \
    relevance_folder_path + relevance_model_name + '/'
relevant_pair_path = relevance_folder_path + 'pairs.csv'
influence_model_path = \
    influence_folder_path + influence_model_name + '/'
train_data_file = '../influence_data/influence_data.csv'
whole_data_file = '../influence_data/tuples.csv'
token_source_path = '../intermediate_data/breast_cancer.csv'
token_file_path = influence_model_path + 'tokenizer.pickle'
train_idxs_path = influence_folder_path + 'train_idxs.txt'

# parameters
data_type = "influence_classification"
maxlen = 300
batch_size = 32
embed_dim = 128
kernel_size = 3
batch_size = 32
pool_size = 2
filter_num = 32
dense_size = 32
dropout_rate = 0.25
relevance_nb_epoch = 5
influence_nb_epoch = 15
train_ratio = 0.95
max_features = 20000
maxlen = 300
output_dim = 2
threshold = 0.5

if influence_model_name == 'baseline_avg':
    threshold = 0.125
elif influence_model_name == 'baseline_max':
    threshold = 0.2375

# load helpers
pp = preprocessing()
mu = model_util()
pa = result_analysis()

# properties
token_source_columns = [
    'init_post', 
    'replies'
]
influence_columns = [
    'initial_post', 
    'reply_post', 
    'initial_author_reply'
]
model_mapping = {
    'new_arci' : \
        mu.get_arci(
            max_features,
            embed_dim,
            maxlen,
            filter_num,
            kernel_size,
            pool_size,
            dropout_rate,
            dense_size,
            output_dim
        ),
    'new_match_pyramid' : \
        mu.get_match_pyramid(
            maxlen,
            max_features,
            embed_dim,
            dropout_rate
        ),
    'new_dot_match_pyramid' : \
        mu.dot_arci(
            dense_size,
            dropout_rate,
            output_dim
        ),
    'new_dot_arci' : \
        mu.dot_arci(
            dense_size,
            dropout_rate,
            output_dim
        ),
    'new_cat_match_pyramid' : \
        mu.cat_arci(
            dense_size,
            dropout_rate,
            output_dim
        ),
    'new_cat_arci' : \
        mu.cat_arci(
            dense_size,
            dropout_rate,
            output_dim
        ),
    'simple_arci' : \
        mu.simple_arci(
            dense_size, 
            dropout_rate, 
            output_dim
        ),
    'simple_match_pyramid' : \
        mu.simple_arci(
            dense_size,
            dropout_rate,
            output_dim
        )
}

from numpy.random import seed
seed(10)
import tensorflow
tensorflow.random.set_seed(20)
    
# train or load choices
gen_new_input = False
train_relevance = False
train_influence = False

# create directory
pp.create_dirs([
    relevance_model_path,
    influence_model_path
])

# load data
influence_data = pd.read_csv(train_data_file)

# preprocess training data
# calculate question and action probability
question, action = pp.get_question_action(influence_data)
influence_data = pp.prepare_data(
    influence_data, 
    influence_columns
)
tokenizer = pp.get_token(
    token_source_path, 
    token_file_path, 
    token_source_columns,
    max_features
)
posts = pp.get_seq_input(
    influence_data, 
    tokenizer, 
    influence_columns,
    maxlen
)

# get train indices by random sample
if gen_new_input:
    train_idxs = set(
        random.sample(
            range(0, len(influence_data)), 
            int(len(influence_data) * train_ratio)
        )
    )
    # save generated index
    pp.save_data(train_idxs, train_idxs_path)
    # generate relevant pairs from train and test data and save the pairs
    relevance_train_data, relevance_test_data, \
    relevance_train_Y, relevance_test_Y \
    = pp.get_relevant_pairs(
        influence_data, 
        posts, 
        train_idxs, 
        relevant_pair_path
    )
else:
    train_idxs = set(
        pp.load_idx(train_idxs_path)
    )
    # load generated data
    relevance_train_data, relevance_test_data, \
    relevance_train_Y, relevance_test_Y \
    = pp.load_relevant_pairs(relevant_pair_path, maxlen)
    
relevance_train_data, relevance_test_data, \
relevance_train_Y \
= pp.gen_input_to_model(
    relevance_train_data, 
    relevance_test_data, 
    relevance_train_Y
)

if train_relevance:
    # build and train relevance model
    relevance_model = model_mapping[relevance_model_name]
    relevance_model.fit(
        relevance_train_data, 
        relevance_train_Y, 
        epochs=relevance_nb_epoch, 
        batch_size=batch_size, 
        verbose=output_dim
    )
    # save relevance model
    pa.save_model(
        relevance_model, 
        relevance_model_path
    )
else:
    # load trained relevance model
    relevance_model = pp.load_model(relevance_model_path)

# predict relevance on test data
y_p = relevance_model.predict(
    relevance_test_data, 
    batch_size=batch_size, 
    verbose=output_dim
)
res = [val[0] for val in y_p]

# evaluate relevance results
pa.evaluate_results(y_p, relevance_test_Y, res, threshold)

if influence_model_name != 'baseline_avg' and \
    influence_model_name != 'baseline_max':
    # remove the softmax layer
    relevance_model = mu.remove_softmax_layer(relevance_model)

# generate influence input
train12, train23, \
test12, test23, \
train_question, test_question, \
train_action, test_action, \
train_Y, test_Y \
= pp.get_influence_input(
    relevance_model, 
    posts, 
    influence_data['label'].values, 
    question,
    action,
    train_idxs,
    batch_size,
    output_dim
)

if not influence_model_name.startswith('baseline'):
    if train_influence:
        # build and train influence model
        influence_model = model_mapping[influence_model_name]
        train_input = []
        if influence_model_name.startswith('simple'):
            train_input = [train12, train23]
        else:
            train_input = [
                train12,
                train23,
                train_question,
                train_action
            ]
        influence_model.fit(
            train_input, 
            train_Y, 
            epochs=influence_nb_epoch, 
            batch_size=batch_size, 
            verbose=output_dim
        )
        # save influence model
        pa.save_model(
            influence_model, 
            influence_model_path
        )
    else:
        # load trained influence model
        influence_model = pp.load_model(influence_model_path)
        print(
            'Train data percentage: {}'.format(
                sum([v[0] for v in train_Y]) / len(train_Y)
            )
        )
    test_input = []
    if influence_model_name.startswith('simple'):
        test_input = [test12, test23]
    else:
        test_input = [
            test12,
            test23,
            test_question,
            test_action
        ]
    # predict influence on test data
    y_p = influence_model.predict(
        test_input, 
        batch_size=batch_size, 
        verbose=output_dim
    )
else:
    y_p = mu.baseline_predict(
        test12, 
        test23, 
        test_question, 
        test_action,
        influence_model_name
    )

res = [val[0] for val in y_p]

# evaluate influence results
fpr, tpr = pa.evaluate_results(
        y_p, 
        test_Y, 
        res, 
        threshold, 
        test_question, 
        test_action,
        check_question_action=False
    )
