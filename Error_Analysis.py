########################################################################################################################
# Error Analysis -------------------------------------------------------------------------------------------------------
########################################################################################################################
import numpy as np
import os
import json
import glob
import matplotlib.pyplot as plt

def plot_errorDistribution(errors_dict,task):
    # Prepare data for plotting
    categories = list(errors_dict.keys())
    occurrences = [len(values) for values in errors_dict.values()]

    # Create a bar chart
    plt.figure(figsize=(10, 8))  # Adjust the figure size as needed
    plt.barh(categories,occurrences,color='firebrick')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Error Categories')
    plt.title('Error Category Occurrences: ' + task)
    plt.xticks(rotation=45)
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels

    # Show the plot
    plt.show()

def sort_rows_descending(array): # Higher value on 4th
    for col in range(array.shape[1]):
        if array[4, col] < array[5, col]:  # If value in 5th row is higher than in 4th row
            # Swap values in 4th and 5th rows
            array[4, col], array[5, col] = array[5, col], array[4, col]
            # Swap corresponding values in 2nd and 3rd rows
            array[2, col], array[3, col] = array[3, col], array[2, col]
    return array

def sort_rows_ascending(array): # Higher value on 3th
    for col in range(array.shape[1]):
        if array[4, col] > array[5, col]:  # If value in 5th row is higher than in 4th row
            # Swap values in 4th and 5th rows
            array[4, col], array[5, col] = array[5, col], array[4, col]
            # Swap corresponding values in 2nd and 3rd rows
            array[2, col], array[3, col] = array[3, col], array[2, col]
    return array


########################################################################################################################
# Decision Making ------------------------------------------------------------------------------------------------------
########################################################################################################################
def get_errors_DM(Response, errors_dict, distract_dict, opposite_dict, strength_dict):

    for i in range(Response.shape[1]):
        # Choose task
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(
                correctResponse) \
                else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                else 'distractOrtho'

            errorComponent_2 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                else 'responseNone' if participantResponse == 'NoResponse' \
                else 'responseOrtho'

            strengthDiff =  strength_dict.get(distractStim.split('_')[0], 0) - strength_dict.get(correctStim.split('_')[0], 0)
            errorComponent_3 = f'strengthDiff{int(strengthDiff * 100)}'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

            # Dynamically add to errors_dict
            if currentChosenList not in errors_dict:
                errors_dict[currentChosenList] = []
            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# Define dicts
distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R'}
opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}
strength_dict = {'lowest':0.25, 'low':0.5, 'strong':0.75, 'strongest':1.0}

# DM -------------------------------------------------------------------------------------------------------------------
errors_dict_DM = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_01\\PreprocessedData_wResp_01\\DM', '*Response.npy'))

for npy_file in npy_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that HIGHER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_descending(Response)
    errors_dict_DM = get_errors_DM(sortedResponse, errors_dict_DM, distract_dict, opposite_dict, strength_dict)
# Visualize results
plot_errorDistribution(errors_dict_DM,'DM')

# DM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_DM_Anti = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_01\\PreprocessedData_wResp_01\\DM_Anti', '*Response.npy'))

for npy_file in npy_files:
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_DM_Anti = get_errors_DM(sortedResponse, errors_dict_DM_Anti, distract_dict, opposite_dict, strength_dict)
# Visualize results
plot_errorDistribution(errors_dict_DM_Anti,'DM_Anti')


########################################################################################################################
# Executive Function ---------------------------------------------------------------------------------------------------
########################################################################################################################
# Define dicts
distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R'}
opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}
def get_errors_EF(Response, errors_dict, distract_dict, opposite_dict):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(
                correctResponse) \
                else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                else 'distractOrtho'

            errorComponent_2 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                else 'responseNone' if participantResponse == 'NoResponse' \
                else 'responseOrtho'

            errorComponent_3 =  distractStim.split('_')[0] + 'Distract_' + correctStim.split('_')[0] + 'Center'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

            # Dynamically add to errors_dict
            if currentChosenList not in errors_dict:
                errors_dict[currentChosenList] = []
            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# EF -------------------------------------------------------------------------------------------------------------------
errors_dict_EF = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_01\\PreprocessedData_wResp_01\\EF', '*Response.npy'))

for npy_file in npy_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF = get_errors_EF(sortedResponse, errors_dict_EF, distract_dict, opposite_dict)
# Visualize results
plot_errorDistribution(errors_dict_EF,'EF')

# EF Anti --------------------------------------------------------------------------------------------------------------
errors_dict_EF_Anti = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_01\\PreprocessedData_wResp_01\\EF_Anti', '*Response.npy'))

for npy_file in npy_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF_Anti = get_errors_EF(Response, errors_dict_EF, distract_dict, opposite_dict)
# Visualize results
plot_errorDistribution(errors_dict_EF_Anti,'EF_Anti')


########################################################################################################################
# Relational Processing ------------------------------------------------------------------------------------------------
########################################################################################################################
# Define dicts
distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R'}
opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}

def sort_rows_RP(array):
    # look for postion of correctResponse in array
    for col in range(array.shape[1]):
        for i, v in enumerate(array[2:, col]):
            if v == array[1, col]:
                if i+2 != 2: # +2 because we enumerate in array from 2:
                    # Swap values
                    array[2, col], array[i+2, col] = array[i+2, col], array[2, col]
                    # Swap corresponding number values
                    array[7, col], array[i+7, col] = array[i+7, col], array[7, col]
    return array

def get_errors_RP(Response, errors_dict, opened_meta_file):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            errorComponent_1 = 'distractColor' + '-' + distractStim.split('_')[0]

            errorComponent_2 = 'distractForm' + '-' + distractStim.split('_')[1].split('.')[0]

            errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials')[1]

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}{errorComponent_3}'

            # Dynamically add to errors_dict
            if currentChosenList not in errors_dict:
                errors_dict[currentChosenList] = []
            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# RP -------------------------------------------------------------------------------------------------------------------
errors_dict_RP = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_RP(Response)
    errors_dict_RP = get_errors_RP(sortedResponse, errors_dict_RP, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_RP,'RP')

# RP Anti --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Anti = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Anti', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Anti', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_RP(Response)
    errors_dict_RP_Anti = get_errors_RP(sortedResponse, errors_dict_RP_Anti, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_RP_Anti,'RP_Anti')

# RP Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx1 = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Ctx1', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Ctx1', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_RP(Response)
    errors_dict_RP_Ctx1 = get_errors_RP(sortedResponse, errors_dict_RP_Ctx1, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_RP_Ctx1,'RP_Ctx1')

# RP Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx2 = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Ctx2', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\RP_Ctx2', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_RP(Response)
    errors_dict_RP_Ctx2 = get_errors_RP(sortedResponse, errors_dict_RP_Ctx2, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_RP_Ctx2,'RP_Ctx2')


########################################################################################################################
# Working Memory -------------------------------------------------------------------------------------------------------
########################################################################################################################
def sort_rows_WM(array):
    # look for postion of correctResponse in array
    for col in range(array.shape[1]):
            if array[2, col] != array[1, col]: # +2 because we enumerate in array from 2:
                # Swap values
                array[2, col], array[3, col] = array[3, col], array[2, col]
    return array

def get_errors_WM(Response, errors_dict, opened_meta_file):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            errorComponent_1 = 'distractColor' + '-' + distractStim.split('_')[0]

            errorComponent_2 = 'distractForm' + '-' + distractStim.split('_')[1].split('.')[0]

            errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials')[1]

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}{errorComponent_3}'

            # Dynamically add to errors_dict
            if currentChosenList not in errors_dict:
                errors_dict[currentChosenList] = []
            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# WM -------------------------------------------------------------------------------------------------------------------
errors_dict_WM = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_WM(Response)
    errors_dict_WM = get_errors_WM(sortedResponse, errors_dict_WM, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_WM, 'WM')

# WM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Anti = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Anti', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Anti', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_WM(Response)
    errors_dict_WM_Anti = get_errors_WM(sortedResponse, errors_dict_WM_Anti, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_WM_Anti, 'WM_Anti')

# WM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Anti = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Anti', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Anti', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_WM(Response)
    errors_dict_WM_Anti = get_errors_WM(sortedResponse, errors_dict_WM_Anti, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_WM_Anti, 'WM_Ctx1')

# WM Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx1 = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Ctx1', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Ctx1', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_WM(Response)
    errors_dict_WM_Ctx1 = get_errors_WM(sortedResponse, errors_dict_WM_Ctx1, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_WM_Ctx1, 'WM_Ctx1')

# WM Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx2 = {}
npy_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Ctx2', '*Response.npy'))
meta_files = glob.glob(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_02\\PreprocessedData_wResp_02\\WM_Ctx2', '*Meta.json'))

for npy_file, meta_file in zip(npy_files, meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 2th and 3th row, so that the 2th row is equal to the 1st
    sortedResponse = sort_rows_WM(Response)
    errors_dict_WM_Ctx2 = get_errors_WM(sortedResponse, errors_dict_WM_Ctx2, opened_meta_file)
# Visualize results
plot_errorDistribution(errors_dict_WM_Ctx2, 'WM_Ctx2')


