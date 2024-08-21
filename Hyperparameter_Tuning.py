########################################################################################################################
# info: Hyperparameter Tuning
########################################################################################################################
# Random Grid Search of different hyperparameter sets for automated accumulated model training.

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import time
import random
# from sklearn.model_selection import ParameterGrid
import itertools

import Training

########################################################################################################################
# Create HP combinations and randomly choose a selection
########################################################################################################################
def create_param_combinations(param_grid, sample_size):
    # Create all possible combinations of parameters
    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    # Randomly sample the specified number of combinations
    sampled_combinations = random.sample(all_combinations, sample_size)

    return sampled_combinations

# Info: After first HPs the most probable space inheriting the best solution decreased to the following
best_params = {
    'batch_size': 40,
    'in_type': 'normal',
    'rnn_type': 'LeakyGRU', # additional gating mechanisms should lead to better performance
    'use_separate_input': False,
    'loss_type': 'lsq',
    'optimizer': 'adam',
    'activation': 'relu', # fast and effective convergence
    'tau': 100,
    'dt': 10,
    'sigma_rec': 0.01, # good balance between stability and stochasticity
    'sigma_x': 0.001, # good balance between stability and stochasticity
    'w_rec_init': 'randortho', # helps maintaining the gradient norm and avoid issues like exploding and vanishing gradient
    'l1_h': 0,
    'l2_h': [0.00001, 0.00003, 0.00005], # l2 > l1 regularization as it distributes penalty over all parameters, better for complex behavior to be learned
    'l1_weight': 0,
    'l2_weight': [0.0001, 0.0003, 0.0005],
    'l2_weight_init': [0.0001, 0.0003, 0.0005],
    'p_weight_train': None,
    'learning_rate': 0.0001,
    'n_rnn': [256,512], # good balance between computational costs and size to learn complex behavior (the more parameters the more complex behavior can be learned)
    'c_mask_responseValue': [2.,3.,5.], # weights the important part of the sequence (resonse epoch) more than the rest (fixation epoch)
    'monthsConsidered': ['2', '3', '4', '5', '6', '7', '8'] # more data - better generalizability (if data consistent over time)
}

# Randomly sample a chosen selection combinations
sampled_combinations = create_param_combinations(best_params, 50)


# Training #############################################################################################################
model_number = 1
# Example iteration through the grid
for params in sampled_combinations:
    print('START TRAINING FOR NEW MODEL')
    print(params) # Double check with model output files

    # Predefine certain variables
    participant = 'BeRNN_01'
    monthsConsidered = params['monthsConsidered']

    # Data paths for different server
    # preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\Data', participant,'PreprocessedData_wResp_ALL')
    # preprocessedData_path = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/Data', participant,'PreprocessedData_wResp_ALL')
    preprocessedData_path = os.path.join('/zi/flstorage/group_csp/analyses/oliver.frank/Data/', participant,'PreprocessedData_wResp_ALL')

    model = 'Model_' + str(model_number) + '_' + participant + '_Month_' + monthsConsidered[0] + '-' + monthsConsidered[-1]
    # Model directories for different server
    # model_dir = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_01_HPT01', model)
    model_dir = os.path.join('/zi/flstorage/group_csp/analyses/oliver.frank/BeRNN_models/BeRNN_02_HPT01', model)
    # model_dir = os.path.join('/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main/BeRNN_03_HPT01', model)

    # Define probability of each task being trained
    rule_prob_map = {"DM": 1, "DM_Anti": 1, "EF": 1, "EF_Anti": 1, "RP": 1, "RP_Anti": 1, "RP_Ctx1": 1, "RP_Ctx2": 1,
                     "WM": 1, "WM_Anti": 1, "WM_Ctx1": 1, "WM_Ctx2": 1}

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Measure the training time
    start_time = time.time()
    print(f'START TRAINING MODEL: {model_number}')

    # Split the data ---------------------------------------------------------------------------------------------------
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    # Function to split the files
    def split_files(files, split_ratio=0.8):
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        return files[:split_index], files[split_index:]

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_triplets = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # III: Exclude files with specific substrings in their names
                if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                    continue
                # Include only files that contain any of the months in monthsConsidered
                if not any(month in file for month in monthsConsidered):
                    continue
                base_name = file.split('Input')[0]
                # print(base_name)
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                file_triplets.append((input_file, yloc_file, output_file))

        # Split the file triplets
        train_files, eval_files = split_files(file_triplets)

        # Store the results in the dictionaries
        train_data[subdir] = train_files
        eval_data[subdir] = eval_files

    try:
        Training.train(model_dir=model_dir, hp=params, rule_prob_map=rule_prob_map, train_data = train_data, eval_data = eval_data)

        end_time = time.time()
        elapsed_time_minutes = end_time - start_time / 60
        elapsed_time_hours = elapsed_time_minutes / 60

        print(f"TIME TAKEN TO TRAIN MODEL {model_number}: {elapsed_time_hours:.2f} hours")
    except:
        print("An exception occurred with model number: ", model_number)

    # Count up for next model
    model_number += 1


