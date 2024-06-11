# # Function for sorting out already augmented files #####################################################################
# import os
# import datetime
#
# def get_creation_time(file_path):
#     # Get the creation time of the file
#     if os.name == 'nt':  # For Windows
#         creation_time = os.path.getctime(file_path)
#     else:  # For Unix-like systems
#         stat = os.stat(file_path)
#         try:
#             creation_time = stat.st_birthtime
#         except AttributeError:
#             # For Linux, use last metadata change time as creation time is not available
#             creation_time = stat.st_mtime
#     return datetime.datetime.fromtimestamp(creation_time)
#
# def is_file_newer_than(file_path, date_time):
#     creation_time = get_creation_time(file_path)
#     return creation_time > date_time
#
# # Example usage
# file_path = 'path/to/your/file.txt'
# date_time = datetime.datetime(2023, 5, 1, 12, 0, 0)  # Replace with your specific date and time
#
# if is_file_newer_than(file_path, date_time):
#     print(f"The file {file_path} is newer than {date_time}.")
#     # Your commands here
# else:
#     print(f"The file {file_path} is older than {date_time}.")
#     # Your other commands here
#
#



# Error_Comparison #####################################################################################################
# Define necessary variables at

########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################
from sklearn.model_selection import ParameterGrid

param_grid = {
    'batch_size_train': [16, 32, 64, 128],
    'batch_size_test': [320, 640, 1280],
    'in_type': ['normal', 'multi'],
    'rnn_type': ['NonRecurrent', 'LeakyRNN', 'LeakyGRU', 'EILeakyGRU', 'GRU', 'LSTM'],
    'use_separate_input': [True, False],
    'loss_type': ['lsq', 'cross_entropy', 'huber'],
    'optimizer': ['adam', 'sgd', 'rmsprop', 'adamw'],
    'activation': ['relu', 'softplus', 'tanh', 'elu', 'linear'],
    'tau': [50, 100, 200],
    'dt': [10, 20, 50],
    'sigma_rec': [0.01, 0.05, 0.1],
    'sigma_x': [0.001, 0.01, 0.05],
    'w_rec_init': ['diag', 'randortho', 'randgauss'],
    'l1_h': [0, 0.0001, 0.001],
    'l2_h': [0, 0.00001, 0.0001],
    'l1_weight': [0, 0.0001, 0.001],
    'l2_weight': [0, 0.0001, 0.001],
    'l2_weight_init': [0, 0.0001, 0.001],
    'p_weight_train': [None, 0.05, 0.1],
    'learning_rate': [0.0001, 0.001, 0.01],
    'n_rnn': [128, 256, 512]
}

grid = list(ParameterGrid(param_grid))

# Example iteration through the grid
for params in grid:
    # Update your hp dictionary with the current parameters
    hp.update(params)
    # Train your model with the current hp
    train_model(hp)