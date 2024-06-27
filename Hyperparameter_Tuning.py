# HP Tuning ############################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
from sklearn.model_selection import ParameterGrid

import Training

param_grid = {
    'batch_size': [20, 40],
    'in_type': ['normal'],
    'rnn_type': ['LeakyRNN', 'LeakyGRU', 'GRU', 'LSTM'],
    'use_separate_input': [True, False],
    'loss_type': ['lsq', 'cross_entropy'],
    'optimizer': ['adam'],
    'activation': ['relu', 'softplus', 'tanh'],
    'tau': [50, 100],
    'dt': [10, 20],
    'sigma_rec': [0.01, 0.05],
    'sigma_x': [0.001, 0.01],
    'w_rec_init': ['diag', 'randortho'],
    'l1_h': [0, 0.0001],
    'l2_h': [0, 0.00001],
    'l1_weight': [0, 0.0001],
    'l2_weight': [0, 0.0001],
    'l2_weight_init': [0, 0.0001],
    'p_weight_train': [None, 0.05],
    'learning_rate': [0.0001, 0.001],
    'n_rnn': [128, 256]
}
# Create all possible combinations
grid = list(ParameterGrid(param_grid))
# Randomly sample 100 of them
sampled_grid = random.sample(grid, 2)

# Training #############################################################################################################
model_number = 146
# Example iteration through the grid
for params in sampled_grid:
    print(params) # Double check with model output files
    # Predefine certain variables
    participant = 'BeRNN_01'
    monthsConsidered = ['2', '3', '4', '5', '6']
    preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Data', participant,'PreprocessedData_wResp_ALL')
    # Define probability of each task being trained
    rule_prob_map = {"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1, "RP_Ctx2": 1,
                     "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1}
    model = 'Model_' + str(model_number) + '_' + participant + '_Month_' + monthsConsidered[0] + '-' + monthsConsidered[-1]
    model_dir = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models', model)
    # Count up for next model
    model_number += 1

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Measure the training time
    start_time = time.time()
    print(f'START TRAINING MODEL: {model_number}')

    Training.train(model_dir=model_dir, trial_dir=preprocessedData_path, monthsConsidered=monthsConsidered,hp=params,rule_prob_map=rule_prob_map)

    end_time = time.time()
    elapsed_time = end_time - start_time / 60

    print(f"TIME TAKEN TO TRAIN MODEL {model_number}: {elapsed_time:.2f} minutes")

