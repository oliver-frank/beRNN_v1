# head: ################################################################################################################
# head: Get best model for RNN/GRU/Multi and train 20x, respectively ###################################################
# head: ################################################################################################################
import numpy as np
from pathlib import Path
import os
import time
import ast
import json

from training import train
import tools

# Choose machine to process on
machineList = ['local', 'hitkip']
machine = machineList[0]
# Choose model class
modelClassList = ['RNN', 'multiRNN', 'GRU']
modelClass = modelClassList[0]
# Define performance range
modelRangeList = ['best', 'worst']
modelRange = modelRangeList[0]
# Define data format
dataFormatList = ['highDim', 'highDim_correctOnly', 'highDim3stimTC']
dataFormat = dataFormatList[1]
# Define participant
participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
participant = participantList[1]

# # info: Change the directories for hitkip run ##########################################################################
# participant = 'beRNN_05'
# data = 'correctOnly'
# seperator = 'CorrectOnly'
# # ---- Load the original list (not line-by-line!) ----
# with open(fr"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\paperPlanes\highDim_{data}\{participant}\visuals\performance_test\bestModels_performance_test.txt", "r") as f:
#     old_paths = json.load(f)  # loads full list
#
# # ---- Transform paths ----
# new_paths = []
# for path in old_paths:
#     # Optional: ensure slashes are consistent
#     path = path.replace("\\", "/")
#
#     # Replace Windows base path with Linux path
#     suffix = path.split(f"/highDim_{seperator}/")[1]
#     new_path = f"/zi/home/oliver.frank/Desktop/beRNNmodels/finalGridSearch_allSubjects_{data}/{suffix}"
#     new_paths.append(new_path)
#
# # ---- Save as proper list again (as .txt with JSON structure) ----
# with open(fr"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\paperPlanes\highDim_{data}\{participant}\visuals\performance_test\bestModels_performance_test_hitkipVersion.txt", "w") as f:
#     json.dump(new_paths, f, indent=2)  # indent for readability
# # info: ################################################################################################################


if machine == 'local':
    directory = Path(
        f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/robustnessTest/{dataFormat}/{participant}/visuals/performance_test/bestModels_performance_test.txt')
elif machine == 'hitkip':
    directory = Path(
        '/zi/home/oliver.frank/Desktop/RNN/multitask_BeRNN-main/bestModels_performance_test_hitkipVersion.txt')

with open(directory, 'r') as f:
    content = f.read()

modelList = ast.literal_eval(content)

if modelRange == 'best':
    modelList = modelList
elif modelRange == 'worst':
    modelList = modelList[::-1]

filtered_modelList = []
numberOfModelsToFilter = 20
numberOfCurrentModelFiltered = 0

if modelClass == 'RNN':
    # Get LeakyRNN models for defined range
    for model_ in modelList:  # info: Just start at different point to get best/worst - modelList, modelList[::-1]
        # print(model_)
        if 'LeakyRNN' in model_ and len(Path(model_).parts[-2].split('_')[-1].split('-')) == 1:
            model_dir = model_
            filtered_modelList.append(model_)

            numberOfCurrentModelFiltered += 1

            if len(filtered_modelList) == numberOfModelsToFilter:
                break

if modelClass == 'multiRNN':
    # Get best test performing MultiRNN
    for model_ in modelList:
        # print(model_)
        if 'LeakyRNN' in model_ and len(Path(model_).parts[-2].split('_')[-1].split('-')) > 1:
            model_dir = model_
            filtered_modelList.append(model_)

            numberOfCurrentModelFiltered += 1

            if len(filtered_modelList) == numberOfModelsToFilter:
                break

if modelClass == 'GRU':
    # Get best test performing GRU
    for model_ in modelList:
        # print(model_)
        if 'GRU' in model_:
            model_dir = model_
            filtered_modelList.append(model_)

            numberOfCurrentModelFiltered += 1

            if len(filtered_modelList) == numberOfModelsToFilter:
                break

modelBatch = 0 # Create a new modelBatch for each chosen model in robustnessTest folder
model_dir = filtered_modelList[modelBatch]

print(f'chosen model nr. {numberOfCurrentModelFiltered} : ', model_dir)

# Initialize list for all training times for each model
trainingTimeList = []
for modelNumber in range(1, 5):  # Define number of iterations and models to be created for every month, respectively

    # Measure time for every model, respectively
    trainingTimeTotal_hours = 0
    # Start it
    start_time = time.perf_counter()
    print(f'START TRAINING MODEL: {modelNumber}')

    hp = tools.load_hp(model_dir)
    hp['rng'] = np.random.default_rng()  # info: has to be defined again because of changes in training.py
    load_dir = None

    if machine == 'local':
        path_ = Path('C:/Users/oliver.frank/Desktop/PyProjects')
    elif machine == 'hitkip':
        path_ = Path('/zi/home/oliver.frank/Desktop')

    # Define data path
    preprocessedData_path = os.path.join(path_, 'Data', hp['participant'], hp['data'])

    for month in hp['monthsConsidered']:  # attention: You have to delete this if cascade training should be set OFF
        # Adjust variables manually as needed
        model_name = f'model_{month}'

        if hp['multiLayer'] == True:
            hp['rnn_type'] = 'LeakyRNN'  # info: force rnn_type to always be LeakyRNN for multiLayer case
            numberOfLayers = len(hp['n_rnn_per_layer'])

            if numberOfLayers == 2:
                model_dir = os.path.join(
                    f"{path_}/beRNNmodels/robustnessTest/{dataFormat}/{hp['participant']}/{modelBatch}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}",
                    model_name)
            else:
                model_dir = os.path.join(
                    f"{path_}/beRNNmodels/robustnessTest/{dataFormat}/{hp['participant']}/{modelBatch}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}-{hp['n_rnn_per_layer'][2]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}-{hp['activations_per_layer'][2]}",
                    model_name)
        else:
            model_dir = os.path.join(
                f"{path_}/beRNNmodels/robustnessTest/{dataFormat}/{hp['participant']}/{modelBatch}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn']}_{hp['activation']}",
                model_name)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Create train and eval data
        train_data, eval_data = tools.createSplittedDatasets(hp, preprocessedData_path, month)

        # info: If you want to initialize the new model with an old one
        # load_dir = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\beRNNmodels\\2025_03\\sc_mask_final\\beRNN_03_All_3-5_data_highDim_correctOnly_iteration1_LeakyRNN_1000_relu\\model_month_3'
        # Start Training ---------------------------------------------------------------------------------------------------
        train(preprocessedData_path, model_dir=model_dir, train_data=train_data, eval_data=eval_data, hp=hp, load_dir=load_dir)

        # info: If True previous model parameters will be taken to initialize consecutive model, creating sequential training
        if hp['sequenceMode'] == True:
            load_dir = model_dir

    end_time = time.perf_counter()
    # Training time taken into account
    elapsed_time_seconds = end_time - start_time
    elapsed_time_minutes = elapsed_time_seconds / 60
    elapsed_time_hours = elapsed_time_minutes / 60

    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_seconds:.2f} seconds")
    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_minutes:.2f} minutes")
    print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_hours:.2f} hours")

    # Accumulate trainingTime
    trainingTimeList.append(elapsed_time_hours)
    trainingTimeTotal_hours += elapsed_time_hours

# Save training time total and list to folder as a text file
file_path = os.path.join(f"{path_}/beRNNmodels/robustnessTest/{dataFormat}/{hp['participant']}/{modelBatch}",
                         'times.txt')

with open(file_path, 'w') as f:
    f.write(f"training time total (hours): {trainingTimeTotal_hours}\n")
    f.write("training time each individual model (hours):\n")
    for time in trainingTimeList:
        f.write(f"{time}\n")

print(f"Training times saved to {file_path}")




