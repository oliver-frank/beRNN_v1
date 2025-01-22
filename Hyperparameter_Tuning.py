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
import Tools

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

def create_repeated_param_combinations(param_grid, sample_size):
    # Create the single combination of parameters
    keys, values = zip(*param_grid.items())
    single_combination = dict(zip(keys, [v[0] for v in values]))

    # Return the same combination 'sample_size' times
    repeated_combinations = [single_combination for _ in range(sample_size)]

    return repeated_combinations

# Get input and output dimension for network, depending on higDim and lowDim data and ruleset (standard: 'all')
num_ring = Tools.get_num_ring('all')
n_rule = Tools.get_num_rule('all')
n_eachring = 32 # attention: 10 for lowDim - 32 for highDim
n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1 # attention: n_output: n_output = 2 +1 for lowDim; n_output = n_eachring +1 for highDim

# Info: After first HPs the most probable space inheriting the best solution decreased to the following
adjParams = {
    'batch_size': [40, 80, 120],  # low: [80, 120, 160] - high: [40, 80, 120]
    'in_type': ['normal'],
    'rnn_type': ['LeakyRNN'],
    'n_input': [n_input], # number of input units
    'n_output': [n_output], # number of output units
    'use_separate_input': [False],
    'loss_type': ['lsq'],
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu','softplus','elu'],
    'tau': [50, 100, 150],
    'dt': [20],
    'sigma_rec': [0.01, 0.05, 0.1],
    'sigma_x': [0.01],
    'w_rec_init': ['randortho', 'randgauss'],
    'l1_h': [0.00005, 0.0001, 0.0005], # low: [0.0, 0.00005, 0.0001] - high: [0.00005, 0.0001, 0.0005]
    'l2_h': [0.000005, 0.00001, 0.00005], # low: [0.0, 0.000005, 0.00001] - high: [0.000005, 0.00001, 0.00005]
    'l1_weight': [0.00001, 0.00005, 0.0001],
    'l2_weight': [0.00001, 0.00005, 0.0001],
    'l2_weight_init': [0],
    'p_weight_train': [None, 0.05, 0.1],
    'learning_rate': [0.0005, 0.001, 0.0015],  # low: [0.001, 0.002, 0.005] - high: [0.0005, 0.001, 0.0015]
    'n_rnn': [128, 256, 512], # low: [64, 128, 256] - high: [128, 256, 512]
    'c_mask_responseValue': [5., 3., 1.],
    'monthsConsidered': [['3','4','5']], # list of lists
    'monthsString': ['3-5'],
    # 'rule_prob_map': {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}
    'rule_prob_map': [{"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}], # fraction of tasks represented in training data
    'participant': ['beRNN_03'], # Participant to take
    'data': ['data_highDim'], # 'data_highDim' , data_highDim_correctOnly , data_highDim_lowCognition , data_lowDim , data_lowDim_correctOnly , data_lowDim_lowCognition
    'tasksString': ['AllTask'], # tasksTaken
    'sequenceMode': [True] # Decide if models are trained sequentially month-wise
}
# Randomly sample combinations
sampled_combinations = create_param_combinations(adjParams, 50)

# # Create one combination and repeat it according to sample_size
# sampled_repeated_combinations = create_repeated_param_combinations(best_params, 5)


# Training #############################################################################################################
# Example iteration through the grid
for modelNumber, params in enumerate(sampled_combinations): # info: either sampled_combinations OR sampled_repeated_combinations
    print(params)
    print(modelNumber)

    print('START TRAINING FOR NEW MODEL')
    # print(params) # Double check with model output files
    # ATTENTION: ADAPT WHOLE NAMING ETC. ###############################################################################
    load_dir = None

    # Define main path
    # path = 'C:\\Users\\oliver.frank\\Desktop\\BackUp'  # local
    # path = 'W:\\group_csp\\analyses\\oliver.frank' # fl storage
    # path = '/data' # hitkip cluster
    path = '/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main' # pandora

    # Define data path
    preprocessedData_path = os.path.join(path, 'Data', params['participant'], params['data'])

    for month in params['monthsConsidered']:  # attention: You have to delete this if cascade training should be set OFF
        # Adjust variables manually as needed
        model_name = f'model_{month}'

        # Define model_dir for different servers
        # model_dir = os.path.join(f"{path}\\beRNNmodels\\2025_01\\{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_{params['rnn_type']}_{params['n_rnn']}_{params['activation']}_iteration{modelNumber}", model_name) # local
        model_dir = os.path.join(f"{path}/beRNNmodels/2025_01/{params['participant']}_{params['tasksString']}_{params['monthsString']}_{params['data']}_{params['rnn_type']}_{params['n_rnn']}_{params['activation']}_iteration{modelNumber}",model_name)  # pandora

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Measure the training time
        start_time = time.time()
        print(f'START TRAINING MODEL: {modelNumber}')

        # Split the data ---------------------------------------------------------------------------------------------------
        # List of the subdirectories
        subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]

        # Initialize dictionaries to store training and evaluation data
        train_data = {}
        eval_data = {}

        # Function to split the files
        for subdir in subdirs:
            # Collect all file triplets in the current subdirectory
            file_triplets = []
            for file in os.listdir(subdir):
                if file.endswith('Input.npy'):
                    # III: Exclude files with specific substrings in their names
                    # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                    #     continue
                    if month not in file:  # Sort out months which should not be considered
                        continue
                    # Add all necessary files to triplets
                    base_name = file.split('Input')[0]
                    input_file = os.path.join(subdir, base_name + 'Input.npy')
                    yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                    output_file = os.path.join(subdir, base_name + 'Output.npy')
                    file_triplets.append((input_file, yloc_file, output_file))

            # Split the file triplets
            train_files, eval_files = Training.split_files(file_triplets)

            # Store the results in the dictionaries
            train_data[subdir] = train_files
            eval_data[subdir] = eval_files

        try:
            # Start Training ---------------------------------------------------------------------------------------------------
            Training.train(model_dir=model_dir, train_data=train_data, eval_data=eval_data, hp=params, load_dir=load_dir)

            end_time = time.time()
            elapsed_time_minutes = end_time - start_time / 60
            elapsed_time_hours = elapsed_time_minutes / 60

            print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_hours:.2f} hours")
        except:
            print("An exception occurred with model number: ", modelNumber)

            # info: If True previous model parameters will be taken to initialize consecutive model, creating sequential training
            if params['sequenceMode'] == True:
                load_dir = model_dir


