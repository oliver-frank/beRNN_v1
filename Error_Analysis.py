########################################################################################################################
# Error Analysis -------------------------------------------------------------------------------------------------------
########################################################################################################################
import numpy as np
import os
import json
import glob
import itertools
import matplotlib.pyplot as plt

def plot_errorDistribution(errors_dict,directory,task):
    # Prepare data for plotting
    categories = list(errors_dict.keys())
    occurrences = [len(values) for values in errors_dict.values()]
    # Filter out categories with no occurrences for labeling
    labels = [cat if len(errors_dict[cat]) > 0 else '' for cat in categories]

    participant = directory.split('\\')[6] + ' '

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(10, len(categories)*0.5))  # Adjust the figure size as needed
    ax.barh(categories, occurrences, color='firebrick')
    # Set labels and titles
    ax.set_xlabel('Number of Occurrences')
    ax.set_ylabel('Error Categories')
    ax.set_title('Error Category Occurrences: ' + participant + task)
    # Set y-ticks to all categories but only label those with occurrences
    ax.set_yticks(range(len(categories)))  # Ensure there's a tick for each category
    ax.set_yticklabels(labels)  # Apply the labels (with blanks for no occurrences)
    plt.xticks(rotation=45)
    plt.xlim([0, 50])
    # plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.subplots_adjust(left=0.4, right=0.95, bottom=0.05, top=0.95)
    plt.show()
    # Save plot
    plt.savefig(os.path.join(directory.split('PreprocessedData')[0],'ErrorGraphics' ,participant+task+'.png'))

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
            errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(correctResponse) \
                else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                else 'distractOrtho'

            errorComponent_2 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                else 'responseNone' if participantResponse == 'NoResponse' \
                else 'responseOrtho'

            strengthDiff =  abs(strength_dict.get(distractStim.split('_')[0], 0) - strength_dict.get(correctStim.split('_')[0], 0))
            errorComponent_3 = f'strengthDiff{int(strengthDiff * 100)}'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# Define dicts
distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R'}
opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}
strength_dict = {'lowest':0.25, 'low':0.5, 'strong':0.75, 'strongest':1.0}

# Create categorical names
list1 = ['distractOpposite', 'distractSame', 'distractOrtho']
list2 = ['responseOpposite', 'responseNone', 'responseOrtho']
list3 = ['strengthDiff0', 'strengthDiff25', 'strengthDiff5', 'strengthDiff75']
# Generating all combinations of categorical names
categorical_names = ['_'.join(combination) for combination in itertools.product(list1, list2, list3)]

# DM -------------------------------------------------------------------------------------------------------------------
errors_dict_DM = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_01\\PreprocessedData_wResp_ALL\\DM'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))

for npy_file in npy_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that HIGHER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_descending(Response)
    errors_dict_DM = get_errors_DM(sortedResponse, errors_dict_DM, distract_dict, opposite_dict, strength_dict)

########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################

def get_fine_grained_error(sortedResponse, errors_dict_fineGrained):
    for i in range(sortedResponse.shape[1]):
        # Wrongly chosen distraction
        errorComponent_1 = 'distract' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize()
        errorComponent_2 = sortedResponse[3, i].split('_')[0].capitalize()
        # Missed correct stimulus
        errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
        errorComponent_4 = sortedResponse[2, i].split('_')[0].capitalize()
        # Concatenate error components
        currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'

        errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

    return errors_dict_fineGrained

# def fine_grain_error_analysis(list_error_keys, errors_dict):
list1 = ['distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Lowest', 'Low', 'Strong', 'Strongest']
list3 = ['correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Lowest', 'Low', 'Strong', 'Strongest']
# Generating all combinations of categorical names
categorical_names = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Creating dict with created names
errors_dict_fineGrained = {name: [] for name in categorical_names}

list_error_keys = ['distractOrtho_responseOrtho_strengthDiff25', 'distractOrtho_responseOrtho_strengthDiff0']

for j in list_error_keys:
    error_key_values = errors_dict_DM[j]
    sortedResponse = sort_rows_descending(np.column_stack(error_key_values))
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained)

    plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'DM_fineGrained ' + j)

########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################
#
# # Visualize results
# plot_errorDistribution(errors_dict_DM,participantDirectory,'DM')
#
# # DM Anti --------------------------------------------------------------------------------------------------------------
# errors_dict_DM_Anti = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\DM_Anti'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
#
# for npy_file in npy_files:
#     Response = np.load(npy_file, allow_pickle=True)
#     # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
#     sortedResponse = sort_rows_ascending(Response)
#     errors_dict_DM_Anti = get_errors_DM(sortedResponse, errors_dict_DM_Anti, distract_dict, opposite_dict, strength_dict)
# # Visualize results
# plot_errorDistribution(errors_dict_DM_Anti,participantDirectory,'DM_Anti')
#
#
# ########################################################################################################################
# # Executive Function ---------------------------------------------------------------------------------------------------
# ########################################################################################################################
# # Define dicts
# distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R', 'X.png':'X'}
# opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}
#
# # Create categorical names
# list1 = ['distractOpposite', 'distractSame', 'distractOrtho', 'distractX']
# list2 = ['colorsDiff', 'colorsSame']
# list3 = ['responseOpposite', 'responseNone', 'responseOrtho']
#
# # Generating all combinations of categorical names
# categorical_names = ['_'.join(combination) for combination in itertools.product(list1, list2, list3)]
#
# def get_errors_EF(Response, errors_dict, distract_dict, opposite_dict):
#
#     for i in range(Response.shape[1]):
#         participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
#         # Evaluate errors
#         if participantResponse != correctResponse:
#             errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(
#                 correctResponse) \
#                 else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
#                 else 'distractX' if distract_dict.get(distractStim.split('_')[1]) == 'X' \
#                 else 'distractOrtho'
#
#             if distractStim.split('_')[0] == correctStim.split('_')[0]:
#                 errorComponent_2 = 'colorsSame'
#             else:
#                 errorComponent_2 = 'colorsDiff'
#
#             errorComponent_3 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
#                 else 'responseNone' if participantResponse == 'NoResponse' \
#                 else 'responseOrtho'
#
#             # Concatenate error components
#             currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'
#
#             errors_dict[currentChosenList].append(Response[:, i])
#
#     return errors_dict
#
# # EF -------------------------------------------------------------------------------------------------------------------
# errors_dict_EF = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\EF'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
#
# for npy_file in npy_files:
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
#     sortedResponse = sort_rows_ascending(Response)
#     errors_dict_EF = get_errors_EF(sortedResponse, errors_dict_EF, distract_dict, opposite_dict)
# # Visualize results
# plot_errorDistribution(errors_dict_EF,participantDirectory,'EF')
#
# # EF Anti --------------------------------------------------------------------------------------------------------------
# errors_dict_EF_Anti = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\EF_Anti'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
#
# for npy_file in npy_files:
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
#     sortedResponse = sort_rows_ascending(Response)
#     errors_dict_EF_Anti = get_errors_EF(Response, errors_dict_EF_Anti, distract_dict, opposite_dict)
# # Visualize results
# plot_errorDistribution(errors_dict_EF_Anti,participantDirectory,'EF_Anti')
#
#
# ########################################################################################################################
# # Relational Processing ------------------------------------------------------------------------------------------------
# ########################################################################################################################
# # Define dicts
# colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
#              'ClassGreen' : ['green', 'lime', 'moss'],
#              'ClassBlue': ['purple', 'violet', 'blue'],
#              'ClassRed': ['rust', 'red', 'magenta']}
#
# # Create categorical names
# list1 = ['distractClassYellowCircle', 'distractClassYellowNonagon', 'distractClassYellowHeptagon', 'distractClassYellowPentagon', 'distractClassYellowTriangle',\
#          'distractClassBlueCircle', 'distractClassBlueNonagon', 'distractClassBlueHeptagon', 'distractClassBluePentagon', 'distractClassBlueTriangle',\
#          'distractClassRedCircle', 'distractClassRedNonagon', 'distractClassRedHeptagon', 'distractClassRedPentagon', 'distractClassRedTriangle',\
#          'distractClassGreenCircle', 'distractClassGreenNonagon', 'distractClassGreenHeptagon', 'distractClassGreenPentagon', 'distractClassGreenTriangle',\
#          'noResponse']
#
# # Generating all combinations of categorical names
# categorical_names = ['_'.join(combination) for combination in itertools.product(list1)]
#
# def get_errors_RP(Response, errors_dict, opened_meta_file):
#
#     for i in range(Response.shape[1]):
#         participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
#         # Evaluate errors
#         if participantResponse != correctResponse:
#             # Chosen wrong distraction belonging class
#             if Response[0, i].split('_')[0] in colorDict['ClassYellow']:
#                 errorComponent_1 = 'distract' + 'ClassYellow' + participantResponse.split('_')[1].split('.')[0].capitalize()
#             elif Response[0, i].split('_')[0] in colorDict['ClassBlue']:
#                 errorComponent_1 = 'distract' + 'ClassBlue' + participantResponse.split('_')[1].split('.')[0].capitalize()
#             elif Response[0, i].split('_')[0] in colorDict['ClassRed']:
#                 errorComponent_1 = 'distract' + 'ClassRed' + participantResponse.split('_')[1].split('.')[0].capitalize()
#             elif Response[0, i].split('_')[0] in colorDict['ClassGreen']:
#                 errorComponent_1 = 'distract' + 'ClassGreen' + participantResponse.split('_')[1].split('.')[0].capitalize()
#             else:
#                 errorComponent_1 = 'noResponse'
#
#             # # Missed correct class belonging
#             # if Response[1, i].split('_')[0] in colorDict['ClassYellow']:
#             #     if Response[1, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
#             #         errorComponent_2 = 'correct' + 'ClassYellow' + 'Circle'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#             #         errorComponent_2 = 'correct' + 'ClassYellow' + 'Polygon'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#             #         errorComponent_2 = 'correct' + 'ClassYellow' + 'Triangle'
#             # if Response[1, i].split('_')[0] in colorDict['ClassBlue']:
#             #     if Response[1, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
#             #         errorComponent_2 = 'correct' + 'ClassBlue' + 'Circle'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#             #         errorComponent_2 = 'correct' + 'ClassBlue' + 'Polygon'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#             #         errorComponent_2 = 'correct' + 'ClassBlue' + 'Triangle'
#             # if Response[1, i].split('_')[0] in colorDict['ClassRed']:
#             #     if Response[1, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
#             #         errorComponent_2 = 'correct' + 'ClassRed' + 'Circle'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#             #         errorComponent_2 = 'correct' + 'ClassRed' + 'Polygon'
#             #     elif Response[1, i].split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#             #         errorComponent_2 = 'correct' + 'ClassRed' + 'Triangle'
#
#             # errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1].split('_')[1]
#             # Concatenate error components
#             currentChosenList = f'{errorComponent_1}'
#
#             errors_dict[currentChosenList].append(Response[:, i])
#
#     return errors_dict
#
# # RP -------------------------------------------------------------------------------------------------------------------
# errors_dict_RP = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\RP'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_RP = get_errors_RP(Response, errors_dict_RP, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_RP,participantDirectory,'RP')
#
# # RP Anti --------------------------------------------------------------------------------------------------------------
# errors_dict_RP_Anti = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\RP_Anti'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_RP_Anti = get_errors_RP(Response, errors_dict_RP_Anti, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_RP_Anti,participantDirectory,'RP_Anti')
#
# # RP Ctx1 --------------------------------------------------------------------------------------------------------------
# errors_dict_RP_Ctx1 = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\RP_Ctx1'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_RP_Ctx1 = get_errors_RP(Response, errors_dict_RP_Ctx1, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_RP_Ctx1,participantDirectory,'RP_Ctx1')
#
# # RP Ctx2 --------------------------------------------------------------------------------------------------------------
# errors_dict_RP_Ctx2 = {name: [] for name in categorical_names}
# # Get list of necessary files in directory
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\RP_Ctx2'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_RP_Ctx2 = get_errors_RP(Response, errors_dict_RP_Ctx2, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_RP_Ctx2,participantDirectory,'RP_Ctx2 ')
#
#
# ########################################################################################################################
# # Working Memory -------------------------------------------------------------------------------------------------------
# ########################################################################################################################
# colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
#              'ClassGreen' : ['green', 'lime', 'moss'],
#              'ClassBlue': ['purple', 'violet', 'blue'],
#              'ClassRed': ['rust', 'red', 'magenta']}
# formDict = {'ClassCircle': ['circle', 'nonagon'],
#             'ClassPolygon': ['heptagon', 'pentagon'],
#             'ClassTriangle': ['triangle']}
#
# # Create categorical names for WM tasks
# list1 = ['distractClassYellowCircle', 'distractClassYellowPolygon', 'distractClassYellowTriangle',\
#          'distractClassBlueCircle', 'distractClassBluePolygon', 'distractClassBlueTriangle',\
#          'distractClassRedCircle', 'distractClassRedPolygon', 'distractClassRedTriangle',\
#          'distractClassGreenCircle', 'distractClassGreenPolygon', 'distractClassGreenTriangle',\
#          'noResponse']
# list2 = ['diffColor_diffForm', 'simColor_diffForm', 'simColor_simForm']
#
# categorical_names_WM = ['_'.join(combination) for combination in itertools.product(list1, list2)]
#
# # Create categorical names for WM_Ctx task
# list1 = ['formClassCombi_CircleCircle', 'formClassCombi_CirclePolygon', 'formClassCombi_CircleTriangle',\
#          'formClassCombi_PolygonPolygon', 'formClassCombi_PolygonTriangle', 'formClassCombi_TriangleTriangle']
# list2 = ['simColor', 'diffColor']
# list3 = ['simForm', 'diffForm']
# list4 = ['responseMatch', 'responseMismatch', 'responseNoResponse']
#
# categorical_names_WM_Ctx = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
#
# def get_errors_WM(Response, errors_dict, opened_meta_file):
#
#     for i in range(Response.shape[1]):
#         participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
#         # Evaluate errors
#         if participantResponse != correctResponse:
#             # Chosen wrong distraction belonging class
#             if participantResponse.split('_')[0] in colorDict['ClassYellow']:
#                 if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'distract' + 'ClassYellow' + 'Circle'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#                     errorComponent_1 = 'distract' + 'ClassYellow' + 'Polygon'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#                     errorComponent_1 = 'distract' + 'ClassYellow' + 'Triangle'
#             if participantResponse.split('_')[0] in colorDict['ClassBlue']:
#                 if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'distract' + 'ClassBlue' + 'Circle'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#                     errorComponent_1 = 'distract' + 'ClassBlue' + 'Polygon'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#                     errorComponent_1 = 'distract' + 'ClassBlue' + 'Triangle'
#             if participantResponse.split('_')[0] in colorDict['ClassRed']:
#                 if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'distract' + 'ClassRed' + 'Circle'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#                     errorComponent_1 = 'distract' + 'ClassRed' + 'Polygon'
#                 elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#                     errorComponent_1 = 'distract' + 'ClassRed' + 'Triangle'
#             else:
#                 errorComponent_1 = 'noResponse'
#
#             errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1]
#             # Concatenate error components
#             currentChosenList = f'{errorComponent_1}_{errorComponent_2}'
#
#             errors_dict[currentChosenList].append(Response[:, i])
#
#     return errors_dict
#
# def get_errors_WM_Ctx(Response, errors_dict, opened_meta_file):
#
#     for i in range(Response.shape[1]):
#         participantResponse, correctResponse, Stim1, Stim2 = Response[0:4, i]
#         # Evaluate errors
#         if participantResponse != correctResponse:
#             # Find this error's belonging to color class
#             try:
#                 if Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'formClassCombi_CircleCircle'
#                 elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#                     errorComponent_1 = 'formClassCombi_PolygonPolygon'
#                 elif Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
#                     errorComponent_1 = 'formClassCombi_TriangleTriangle'
#                 elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon'] or\
#                         Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'formClassCombi_CirclePolygon'
#                 elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
#                      Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
#                     errorComponent_1 = 'formClassCombi_CircleTriangle'
#                 elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
#                      Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
#                     errorComponent_1 = 'formClassCombi_PolygonTriangle'
#             except Exception as e:
#                 print('Error occured: ', e)
#                 continue
#
#             errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1].split('_')[0]
#
#             errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1].split('_')[1]
#
#             if participantResponse == 'Match' or participantResponse == 'Mismatch':
#                 errorComponent_4 = 'response'+participantResponse
#             else:
#                 errorComponent_4 = 'responseNoResponse'
#
#             # Concatenate error components
#             currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
#
#             errors_dict[currentChosenList].append(Response[:, i])
#
#     return errors_dict
#
# # WM -------------------------------------------------------------------------------------------------------------------
# errors_dict_WM = {name: [] for name in categorical_names_WM}
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\WM'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_WM = get_errors_WM(Response, errors_dict_WM, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_WM, participantDirectory,'WM')
#
# # WM Anti --------------------------------------------------------------------------------------------------------------
# errors_dict_WM_Anti = {name: [] for name in categorical_names_WM}
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\WM_Anti'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_WM_Anti = get_errors_WM(Response, errors_dict_WM_Anti, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_WM_Anti,participantDirectory,'WM_Anti')
#
# # WM Ctx1 --------------------------------------------------------------------------------------------------------------
# errors_dict_WM_Ctx1 = {name: [] for name in categorical_names_WM_Ctx}
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\WM_Ctx1'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_WM_Ctx1 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx1, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_WM_Ctx1,participantDirectory,'WM_Ctx1')
#
# # WM Ctx2 --------------------------------------------------------------------------------------------------------------
# errors_dict_WM_Ctx2 = {name: [] for name in categorical_names_WM_Ctx}
# participantDirectory = 'Z:\\Desktop\\ZI\\PycharmProjects\\BeRNN\\Data\\BeRNN_05\\PreprocessedData_wResp_ALL\\WM_Ctx2'
# npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
#
# for npy_file, meta_file in zip(npy_files, meta_files):
#     # Load the JSON content from the file
#     with open(meta_file, 'r') as file:
#         opened_meta_file = json.load(file)
#     # Use the function
#     Response = np.load(npy_file, allow_pickle=True)
#     errors_dict_WM_Ctx2 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx2, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_WM_Ctx2,participantDirectory,'WM_Ctx2')


