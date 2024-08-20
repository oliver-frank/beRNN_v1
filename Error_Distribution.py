########################################################################################################################
# info: Error Distribution
########################################################################################################################
# Several functions to create a distribution of error classes for the particpant's behavior on the different task.
# The classes to be shown can be manually chosen by the user and can be rough- (few classes) or fine- (many classes)
# grained. For a nice readable overview it is recommended to choose only a few lasses for the fine-grained distribution.

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
import numpy as np
import os
import json
import glob
import itertools
import matplotlib.pyplot as plt

def plot_errorDistribution(errors_dict,directory,task,grainity):
    # Prepare data for plotting
    categories = list(errors_dict.keys())
    occurrences = [len(values) for values in errors_dict.values()]
    # Filter out categories with no occurrences for labeling
    labels = [cat if len(errors_dict[cat]) > 0 else '' for cat in categories]

    participant = directory.split('\\')[5] + ' '

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(50, len(categories)*0.5))  # Adjust the figure size as needed
    ax.barh(categories, occurrences, color='firebrick')
    # Set labels and titles
    ax.set_xlabel('Number of Occurrences')
    ax.set_ylabel('Error Categories')
    ax.set_title('Error Category Occurrences: ' + participant + task)
    # Set y-ticks to all categories but only label those with occurrences
    ax.set_yticks(range(len(categories)))  # Ensure there's a tick for each category
    ax.set_yticklabels(labels)  # Apply the labels (with blanks for no occurrences)
    plt.xticks(rotation=45)
    if grainity == 'rough':
        plt.xlim([0, 300])
    elif grainity == 'fine':
        plt.xlim([0, 60])
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.subplots_adjust(left=0.4, right=0.95, bottom=0.05, top=0.95)
    plt.show()
    # Save plot
    plt.savefig(os.path.join(directory.split('PreprocessedData')[0],'ErrorGraphics' ,participant+task+'.png'), dpi=100)

def plot_errorDistribution_relative(errors_dict, directory, task, grainity):
    # Prepare data for plotting
    categories = list(errors_dict.keys())
    total_occurrences = sum(len(values) for values in errors_dict.values())  # Total occurrences across all categories
    # Calculate relative occurrences (percentages)
    occurrences = [(len(values) / total_occurrences) * 100 for values in errors_dict.values()]
    # Filter out categories with no occurrences for labeling
    labels = [cat if len(errors_dict[cat]) > 0 else '' for cat in categories]

    participant = directory.split('\\')[5] + ' '

    # Create a bar chart
    fig, ax = plt.subplots(figsize=(20, len(categories) * 0.02))  # Adjust the figure size as needed
    ax.barh(categories, occurrences, color='firebrick')
    # Set labels and titles
    ax.set_xlabel('Percentage of Total Occurrences (%)', fontsize = 14)
    ax.set_ylabel('Error Categories', fontsize = 14)
    ax.set_title('Relative Error Category Occurrences: ' + participant + task, fontsize = 16)
    # Set y-ticks to all categories but only label those with occurrences
    ax.set_yticks(range(len(categories)))  # Ensure there's a tick for each category
    ax.set_yticklabels(labels)  # Apply the labels (with blanks for no occurrences)
    plt.xticks(rotation=45)
    if grainity == 'rough':
        plt.xlim([0, 100])  # Adjusted for percentage
    elif grainity == 'fine':
        plt.xlim([0, 100])  # Adjusted for percentage
    plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
    plt.subplots_adjust(left=0.4, right=0.95, bottom=0.05, top=0.95)
    plt.show()
    # Save plot
    plt.savefig(os.path.join(directory.split('PreprocessedData')[0], 'ErrorGraphics', participant + task + '_relative' + '.png'),
                dpi=100)

def sort_rows_descending(array): # Higher value on 4th
    for col in range(array.shape[1]):
        if array[4, col] < array[5, col]:  # If value in 5th row is higher than in 4th row
            # Swap values in 4th and 5th rows
            array[4, col], array[5, col] = array[5, col], array[4, col]
            # Swap corresponding values in 2nd and 3rd rows
            array[2, col], array[3, col] = array[3, col], array[2, col]
    return array

def sort_rows_ascending(array): # Higher value on 5th
    for col in range(array.shape[1]):
        if array[4, col] > array[5, col]:  # If value in 5th row is higher than in 4th row
            # Swap values in 4th and 5th rows
            array[4, col], array[5, col] = array[5, col], array[4, col]
            # Swap corresponding values in 2nd and 3rd rows
            array[2, col], array[3, col] = array[3, col], array[2, col]
    return array

def sort_rows_correctOn2(array): # Correct stim must be on 2nd row
    for col in range(array.shape[1]):
        if array[1, col] != array[2, col]:
            # Swap values in 2th and 3th rows
            array[2, col], array[3, col] = array[3, col], array[2, col]
            # Swap corresponding values in 4th and 5th rows is not necessary
    return array

def get_fine_grained_error(sortedResponse, errors_dict_fineGrained, task):
    for i in range(sortedResponse.shape[1]):
        if task == 'DM' or task == 'DM_Anti':
            # Distraction
            errorComponent_1 = 'distract' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize()
            errorComponent_2 = sortedResponse[3, i].split('_')[0].capitalize()
            # Missed correct stimulus
            errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
            errorComponent_4 = sortedResponse[2, i].split('_')[0].capitalize()
            # Incorrect answer
            errorComponent_5 = 'response' + sortedResponse[0, i].capitalize()
            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

        elif task == 'EF' or task == 'EF_Anti':
            # Distraction
            errorComponent_1 = 'distract' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize()
            errorComponent_2 = sortedResponse[3, i].split('_')[0].capitalize()
            # Missed correct stimulus
            errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
            errorComponent_4 = sortedResponse[2, i].split('_')[0].capitalize()
            # Incorrect answer
            errorComponent_5 = 'response' + sortedResponse[0, i]
            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

        elif task == 'RP' or task == 'RP_Anti' or task == 'RP_Ctx1' or task == 'RP_Ctx2':
            # Wrongly chosen distraction
            if sortedResponse[0, i] == '000_000.png': # errors where the wrong stim was hitten
                continue
            elif sortedResponse[0, i] == 'NoResponse':
                errorComponent_1 = 'distract' + 'NoResponse'
                errorComponent_2 = sortedResponse[1, i].split('_')[0].capitalize()
            else:
                errorComponent_1 = 'distract' + sortedResponse[0, i].split('_')[1].split('.')[0].capitalize()
                errorComponent_2 = sortedResponse[0, i].split('_')[0].capitalize()
            # Missed correct stimulus
            errorComponent_3 = 'correct' + sortedResponse[2, i].split('_')[1].split('.')[0].capitalize()
            if sortedResponse[0, i].split('_')[0] != '000' and sortedResponse[0, i].split('_')[0] != 'NoResponse':
                distractClass = next((cls for cls, colors in colorDict.items() if sortedResponse[0, i].split('_')[0] in colors), None)
                correctClass = next((cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
                if distractClass == correctClass:
                    errorComponent_4 = 'Similiar'
                else:
                    errorComponent_4 = 'NonSimiliar'
            else:
                errorComponent_4 = 'NonSimiliar'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

        elif task == 'WM' or task == 'WM_Anti':
            # Wrongly chosen distraction OR distraction that led to noResponse/empty field
            if sortedResponse[0, i] == '000_000.png':  # errors where the empty field was hitten
                continue
            elif sortedResponse[0, i] == 'noResponse' or sortedResponse[0, i] == 'NoResponse':
                errorComponent_1 = 'noResponse' + sortedResponse[3, i].split('_')[1].split('.')[0].capitalize() # Form
                if sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'ClassYellow'
                elif sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                    errorComponent_2 = 'ClassBlue'
                elif sortedResponse[3, i].split('_')[0] in colorDict['ClassRed']:
                    errorComponent_2 = 'ClassRed'
                elif sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                    errorComponent_2 = 'ClassGreen'
            else:
                errorComponent_1 = 'distract' + sortedResponse[0, i].split('_')[1].split('.')[0].capitalize() # Form
                if sortedResponse[0, i].split('_')[0] in colorDict['ClassYellow']:
                    errorComponent_2 = 'ClassYellow'
                elif sortedResponse[0, i].split('_')[0] in colorDict['ClassBlue']:
                    errorComponent_2 = 'ClassBlue'
                elif sortedResponse[0, i].split('_')[0] in colorDict['ClassRed']:
                    errorComponent_2 = 'ClassRed'
                elif sortedResponse[0, i].split('_')[0] in colorDict['ClassGreen']:
                    errorComponent_2 = 'ClassGreen'
            # Missed correct stimulus
            correctColorClass = next((cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
            distractColorClass = next((cls for cls, colors in colorDict.items() if sortedResponse[3, i].split('_')[0] in colors), None)
            if correctColorClass == distractColorClass:
                errorComponent_3 = 'simColor'
            else:
                errorComponent_3 = 'diffColor'

            correctFormClass = next((cls for cls, forms in formDict.items() if sortedResponse[2, i].split('_')[1].split('.')[0] in forms), None)
            distractFormClass = next((cls for cls, forms in formDict.items() if sortedResponse[3, i].split('_')[1].split('.')[0] in forms), None)
            if correctFormClass == distractFormClass:
                errorComponent_4 = 'simForm'
            else:
                errorComponent_4 = 'diffForm'

            # errorComponent_5 = opened_meta_file['difficultyLevel'].split('trials_')[1]

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])

        elif task == 'WM_Ctx1' or task == 'WM_Ctx2':
            if sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                errorComponent_1 = 'formClassCombi_' + 'CircleCircle'
            elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                errorComponent_1 = 'formClassCombi_' + 'PolygonPolygon'
            elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                errorComponent_1 = 'formClassCombi_' + 'TriangleTriangle'
            elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] or\
                    sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                errorComponent_1 = 'formClassCombi_' + 'CirclePolygon'
            elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassCircle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] or\
                    sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassCircle']:
                errorComponent_1 = 'formClassCombi_' + 'CircleTriangle'
            elif sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassPolygon'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] or\
                    sortedResponse[2, i].split('_')[1].split('.')[0] in formDict['ClassTriangle'] and sortedResponse[3, i].split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                errorComponent_1 = 'formClassCombi_' + 'PolygonTriangle'
            else:
                continue

            if sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                errorComponent_2 = 'colorClassCombi_' + 'YellowYellow'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                errorComponent_2 = 'colorClassCombi_' + 'GreenGreen'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                errorComponent_2 = 'colorClassCombi_' + 'BlueBlue'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed']:
                errorComponent_2 = 'colorClassCombi_' + 'RedRed'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                errorComponent_2 = 'colorClassCombi_' + 'YellowGreen'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                errorComponent_2 = 'colorClassCombi_' + 'YellowBlue'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassYellow'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassYellow']:
                errorComponent_2 = 'colorClassCombi_' + 'YellowRed'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                errorComponent_2 = 'colorClassCombi_' + 'GreenBlue'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassGreen'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassGreen']:
                errorComponent_2 = 'colorClassCombi_' + 'GreenRed'
            elif sortedResponse[2, i].split('_')[0] in colorDict['ClassBlue'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassRed'] or\
                    sortedResponse[2, i].split('_')[0] in colorDict['ClassRed'] and sortedResponse[3, i].split('_')[0] in colorDict['ClassBlue']:
                errorComponent_2 = 'colorClassCombi_' + 'BlueRed'
            else:
                continue

            errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1]

            if sortedResponse[0, i] == 'noResponse' or sortedResponse[0, i] == 'NoResponse':
                errorComponent_4 = 'response' + 'NoResponse'  # Form
            else:
                errorComponent_4 = 'response' + sortedResponse[0, i]

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])
    return errors_dict_fineGrained

focusedMonths = ['month_2','month_3','month_4','month_5','month_6']
directory = 'W:\\group_csp\\analyses\\oliver.frank\\Data\\BeRNN_03\\PreprocessedData_wResp_ALL\\'

########################################################################################################################
# Decision Making
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
participantDirectory = directory + 'DM'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that HIGHER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_descending(Response)
    errors_dict_DM = get_errors_DM(sortedResponse, errors_dict_DM, distract_dict, opposite_dict, strength_dict)
# Visualize results
# plot_errorDistribution(errors_dict_DM,participantDirectory,'DM', 'rough')
# plot_errorDistribution_relative(errors_dict_DM,participantDirectory,'DM', 'rough')
Response = np.load(npy_files[0], allow_pickle=True)

# DM - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Lowest', 'Low', 'Strong', 'Strongest']
list3 = ['correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Lowest', 'Low', 'Strong', 'Strongest']
list5 = ['responseNoResponse', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# info: DM error key pair - has to be added manually
list_error_keys = ['distractOrtho_responseOrtho_strengthDiff25', 'distractOrtho_responseOrtho_strengthDiff0']

for j in list_error_keys:
    error_key_values = errors_dict_DM[j]
    sortedResponse = sort_rows_descending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'DM')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'DM_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'DM_fineGrained ' + j, 'fine')

# DM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_DM_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'DM_Anti'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_DM_Anti = get_errors_DM(sortedResponse, errors_dict_DM_Anti, distract_dict, opposite_dict, strength_dict)
# Visualize results
# plot_errorDistribution(errors_dict_DM_Anti,participantDirectory,'DM_Anti', grainity='rough')
plot_errorDistribution_relative(errors_dict_DM_Anti,participantDirectory,'DM_Anti', grainity='rough')

# DM Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Lowest', 'Low', 'Strong', 'Strongest']
list3 = ['correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Lowest', 'Low', 'Strong', 'Strongest']
list5 = ['responseNoResponse', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# info: DM error key pair - has to be added manually
list_error_keys = ['distractOrtho_responseOrtho_strengthDiff25', 'distractOrtho_responseOrtho_strengthDiff0']

for j in list_error_keys:
    error_key_values = errors_dict_DM_Anti[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'DM_Anti')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'DM_Anti_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'DM_Anti_fineGrained ' + j, 'fine')

########################################################################################################################
# Executive Function
########################################################################################################################
# Define dicts
distract_dict = {'up.png':'U', 'down.png':'D', 'left.png':'L', 'right.png':'R', 'X.png':'X'}
opposite_dict = {'D':'U', 'U':'D', 'R':'L', 'L':'R'}

# Create categorical names
list1 = ['distractOpposite', 'distractSame', 'distractOrtho', 'distractX']
list2 = ['colorsDiff', 'colorsSame']
list3 = ['responseOpposite', 'responseNone', 'responseOrtho']

# Generating all combinations of categorical names
categorical_names = ['_'.join(combination) for combination in itertools.product(list1, list2, list3)]

def get_errors_EF(Response, errors_dict, distract_dict, opposite_dict):

    for i in range(Response.shape[1]):
        if Response[5,i] == '10':
            Response[2,i] = Response[3,i]
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            errorComponent_1 = 'distractOpposite' if distract_dict.get(distractStim.split('_')[1]) == opposite_dict.get(
                correctResponse) \
                else 'distractSame' if distract_dict.get(distractStim.split('_')[1]) == correctResponse \
                else 'distractX' if distract_dict.get(distractStim.split('_')[1]) == 'X' \
                else 'distractOrtho'

            if distractStim.split('_')[0] == correctStim.split('_')[0]:
                errorComponent_2 = 'colorsSame'
            else:
                errorComponent_2 = 'colorsDiff'

            errorComponent_3 = 'responseOpposite' if participantResponse == opposite_dict.get(correctResponse) \
                else 'responseNone' if participantResponse == 'NoResponse' \
                else 'responseOrtho'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# EF -------------------------------------------------------------------------------------------------------------------
errors_dict_EF = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'EF'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF = get_errors_EF(sortedResponse, errors_dict_EF, distract_dict, opposite_dict)
# Visualize results
# plot_errorDistribution(errors_dict_EF,participantDirectory,'EF', grainity='rough')
plot_errorDistribution_relative(errors_dict_EF,participantDirectory,'EF', grainity='rough')

# EF - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractX', 'distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Green', 'Red']
list3 = ['noResponse', 'correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Green', 'Red']
list5 = ['responsenoResponse', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# info: DM error key pair - has to be added manually
list_error_keys = ['distractSame_colorsDiff_responseOrtho', 'distractOpposite_colorsDiff_responseOrtho']

for j in list_error_keys:
    error_key_values = errors_dict_EF[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'EF')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'EF_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'EF_fineGrained ' + j, 'fine')

# EF Anti --------------------------------------------------------------------------------------------------------------
errors_dict_EF_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'EF_Anti'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF_Anti = get_errors_EF(Response, errors_dict_EF_Anti, distract_dict, opposite_dict)
# Visualize results
# plot_errorDistribution(errors_dict_EF_Anti,participantDirectory,'EF_Anti', grainity='rough')
plot_errorDistribution_relative(errors_dict_EF_Anti,participantDirectory,'EF_Anti', grainity='rough')

# EF Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractX', 'distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Green', 'Red']
list3 = ['noResponse', 'correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Green', 'Red']
list5 = ['responsenoResponse', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# info: DM error key pair - has to be added manually
list_error_keys = ['distractSame_colorsSame_responseOrtho', 'distractSame_colorsDiff_responseOrtho']

for j in list_error_keys:
    error_key_values = errors_dict_EF_Anti[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'EF_Anti')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'EF_Anti_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'EF_Anti_fineGrained ' + j, 'fine')

########################################################################################################################
# Relational Processing
########################################################################################################################
# Define dicts
colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
             'ClassGreen' : ['green', 'lime', 'moss'],
             'ClassBlue': ['purple', 'violet', 'blue'],
             'ClassRed': ['rust', 'red', 'magenta']}

# Create categorical names
list1 = ['distractClassYellowCircle', 'distractClassYellowNonagon', 'distractClassYellowHeptagon', 'distractClassYellowPentagon', 'distractClassYellowTriangle',\
         'distractClassBlueCircle', 'distractClassBlueNonagon', 'distractClassBlueHeptagon', 'distractClassBluePentagon', 'distractClassBlueTriangle',\
         'distractClassRedCircle', 'distractClassRedNonagon', 'distractClassRedHeptagon', 'distractClassRedPentagon', 'distractClassRedTriangle',\
         'distractClassGreenCircle', 'distractClassGreenNonagon', 'distractClassGreenHeptagon', 'distractClassGreenPentagon', 'distractClassGreenTriangle',\
         'noResponse']

# Generating all combinations of categorical names
categorical_names = ['_'.join(combination) for combination in itertools.product(list1)]

def get_errors_RP(Response, errors_dict, opened_meta_file):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            # Chosen wrong distraction belonging class
            if Response[0, i].split('_')[0] in colorDict['ClassYellow']:
                errorComponent_1 = 'distract' + 'ClassYellow' + participantResponse.split('_')[1].split('.')[0].capitalize()
            elif Response[0, i].split('_')[0] in colorDict['ClassBlue']:
                errorComponent_1 = 'distract' + 'ClassBlue' + participantResponse.split('_')[1].split('.')[0].capitalize()
            elif Response[0, i].split('_')[0] in colorDict['ClassRed']:
                errorComponent_1 = 'distract' + 'ClassRed' + participantResponse.split('_')[1].split('.')[0].capitalize()
            elif Response[0, i].split('_')[0] in colorDict['ClassGreen']:
                errorComponent_1 = 'distract' + 'ClassGreen' + participantResponse.split('_')[1].split('.')[0].capitalize()
            else:
                errorComponent_1 = 'noResponse'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}'

            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# RP -------------------------------------------------------------------------------------------------------------------
errors_dict_RP = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'RP'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_RP = get_errors_RP(Response, errors_dict_RP, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_RP,participantDirectory,'RP', grainity='rough')
plot_errorDistribution_relative(errors_dict_RP,participantDirectory,'RP', grainity='rough')

# RP - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pair - has to be added manually
list_error_keys = ['noResponse', 'distractClassRedNonagon']

for j in list_error_keys:
    error_key_values = errors_dict_RP[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'RP_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'RP_fineGrained ' + j, 'fine')

# RP Anti --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'RP_Anti'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_RP_Anti = get_errors_RP(Response, errors_dict_RP_Anti, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_RP_Anti,participantDirectory,'RP_Anti', grainity='rough')
plot_errorDistribution_relative(errors_dict_RP_Anti,participantDirectory,'RP_Anti', grainity='rough')

# RP Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pair - has to be added manually
list_error_keys = ['noResponse', 'distractClassBlueNonagon', 'distractClassYellowHeptagon', 'distractClassRedHeptagon']

for j in list_error_keys:
    error_key_values = errors_dict_RP_Anti[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Anti')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'RP_Anti_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'RP_Anti_fineGrained ' + j, 'fine')

# RP Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx1 = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'RP_Ctx1'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_RP_Ctx1 = get_errors_RP(Response, errors_dict_RP_Ctx1, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_RP_Ctx1,participantDirectory,'RP_Ctx1', grainity='rough')
plot_errorDistribution_relative(errors_dict_RP_Ctx1,participantDirectory,'RP_Ctx1', grainity='rough')

# RP Ctx1 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pair - has to be added manually
list_error_keys = ['noResponse']

for j in list_error_keys:
    error_key_values = errors_dict_RP_Ctx1[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Ctx1')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'RP_Ctx1_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'RP_Ctx1_fineGrained ' + j, 'fine')

# RP Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx2 = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = directory + 'RP_Ctx2'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_months_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_RP_Ctx2 = get_errors_RP(Response, errors_dict_RP_Ctx2, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_RP_Ctx2,participantDirectory,'RP_Ctx2 ', grainity='rough')
plot_errorDistribution_relative(errors_dict_RP_Ctx2,participantDirectory,'RP_Ctx2 ', grainity='rough')

# RP Ctx2 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pair - has to be added manually
list_error_keys = ['noResponse', 'distractClassGreenHeptagon', 'distractClassRedNonagon', 'distractClassYellowNonagon']

for j in list_error_keys:
    error_key_values = errors_dict_RP_Ctx2[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Ctx2')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'RP_Ctx2_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'RP_Ctx2_fineGrained ' + j, 'fine')

########################################################################################################################
# Working Memory
########################################################################################################################
colorDict = {'ClassYellow': ['yellow', 'amber', 'orange'],
             'ClassGreen' : ['green', 'lime', 'moss'],
             'ClassBlue': ['purple', 'violet', 'blue'],
             'ClassRed': ['rust', 'red', 'magenta']}
formDict = {'ClassCircle': ['circle', 'nonagon'],
            'ClassPolygon': ['heptagon', 'pentagon'],
            'ClassTriangle': ['triangle']}

# Create categorical names for WM tasks
list1 = ['distractClassYellowCircle', 'distractClassYellowPolygon', 'distractClassYellowTriangle',\
         'distractClassBlueCircle', 'distractClassBluePolygon', 'distractClassBlueTriangle',\
         'distractClassRedCircle', 'distractClassRedPolygon', 'distractClassRedTriangle',\
         'distractClassGreenCircle', 'distractClassGreenPolygon', 'distractClassGreenTriangle',\
         'noResponse']
list2 = ['diffColor_diffForm', 'simColor_diffForm', 'simColor_simForm']

categorical_names_WM = ['_'.join(combination) for combination in itertools.product(list1, list2)]

# Create categorical names for WM_Ctx tasks
list1 = ['formClassCombi_CircleCircle', 'formClassCombi_CirclePolygon', 'formClassCombi_CircleTriangle',\
         'formClassCombi_PolygonPolygon', 'formClassCombi_PolygonTriangle', 'formClassCombi_TriangleTriangle']
list2 = ['diffColor_diffForm', 'simColor_diffForm','diffColor_simForm', 'simColor_simForm']
list3 = ['responseMatch', 'responseMismatch', 'responseNoResponse']

categorical_names_WM_Ctx = ['_'.join(combination) for combination in itertools.product(list1, list2, list3)]

# shows current trial in detail and similiarity in form and color to previous trials
def get_errors_WM(Response, errors_dict, opened_meta_file):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, correctStim, distractStim = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            # Chosen wrong distraction belonging class
            if participantResponse.split('_')[0] in colorDict['ClassYellow']:
                if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'distract' + 'ClassYellow' + 'Circle'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'distract' + 'ClassYellow' + 'Polygon'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'distract' + 'ClassYellow' + 'Triangle'
            elif participantResponse.split('_')[0] in colorDict['ClassGreen']:
                if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'distract' + 'ClassGreen' + 'Circle'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'distract' + 'ClassGreen' + 'Polygon'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'distract' + 'ClassGreen' + 'Triangle'
            elif participantResponse.split('_')[0] in colorDict['ClassBlue']:
                if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'distract' + 'ClassBlue' + 'Circle'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'distract' + 'ClassBlue' + 'Polygon'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'distract' + 'ClassBlue' + 'Triangle'
            elif participantResponse.split('_')[0] in colorDict['ClassRed']:
                if participantResponse.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'distract' + 'ClassRed' + 'Circle'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'distract' + 'ClassRed' + 'Polygon'
                elif participantResponse.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'distract' + 'ClassRed' + 'Triangle'
            elif participantResponse.split('_')[0] == 'noResponse' or participantResponse.split('_')[0] == 'NoResponse':
                errorComponent_1 = 'noResponse'
            else:
                continue

            errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1]
            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}'

            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict
def get_errors_WM_Ctx(Response, errors_dict, opened_meta_file):

    for i in range(Response.shape[1]):
        participantResponse, correctResponse, Stim1, Stim2 = Response[0:4, i]
        # Evaluate errors
        if participantResponse != correctResponse:
            # Find this error's belonging to color class
            try:
                if Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_CircleCircle'
                elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'formClassCombi_PolygonPolygon'
                elif Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle']:
                    errorComponent_1 = 'formClassCombi_TriangleTriangle'
                elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon'] or\
                        Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_CirclePolygon'
                elif Stim1.split('_')[1].split('.')[0] in formDict['ClassCircle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
                     Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassCircle']:
                    errorComponent_1 = 'formClassCombi_CircleTriangle'
                elif Stim1.split('_')[1].split('.')[0] in formDict['ClassPolygon'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassTriangle'] or \
                     Stim1.split('_')[1].split('.')[0] in formDict['ClassTriangle'] and Stim2.split('_')[1].split('.')[0] in formDict['ClassPolygon']:
                    errorComponent_1 = 'formClassCombi_PolygonTriangle'
            except Exception as e:
                print('Error occured: ', e)
                continue

            errorComponent_2 = opened_meta_file['difficultyLevel'].split('trials_')[1]

            if participantResponse == 'Match' or participantResponse == 'Mismatch':
                errorComponent_3 = 'response' + participantResponse
            else:
                errorComponent_3 = 'responseNoResponse'

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}'

            errors_dict[currentChosenList].append(Response[:, i])

    return errors_dict

# WM -------------------------------------------------------------------------------------------------------------------
errors_dict_WM = {name: [] for name in categorical_names_WM}
participantDirectory = directory + 'WM'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the correct stim on the 2nd row
    sortedResponse = sort_rows_correctOn2(Response)
    errors_dict_WM = get_errors_WM(sortedResponse, errors_dict_WM, opened_meta_file)
# # Visualize results
# plot_errorDistribution(errors_dict_WM, participantDirectory,'WM', grainity='rough')
plot_errorDistribution_relative(errors_dict_WM, participantDirectory,'WM', grainity='rough')

# WM - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distract', 'noResponse']
list2 = ['Circle', 'Nonagon', 'Heptagon', 'Pentagon', 'Triangle']
list3 = ['ClassYellow', 'ClassGreen', 'ClassBlue', 'ClassRed']
list4 = ['_simColor_simForm', '_simColor_diffForm', '_diffColor_simForm', '_diffColor_diffForm']
# Generating all combinations of categorical names
categorical_names_fineGrained = [''.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pairs that you want to check out - has to be added manually
list_error_keys = ['noResponse_simColor_simForm']

for j in list_error_keys:
    error_key_values = errors_dict_WM[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'WM_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'WM_fineGrained ' + j, 'fine')

# WM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Anti = {name: [] for name in categorical_names_WM}
participantDirectory = directory + 'WM_Anti'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the correct stim on the 2nd row
    sortedResponse = sort_rows_correctOn2(Response)
    errors_dict_WM_Anti = get_errors_WM(sortedResponse, errors_dict_WM_Anti, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_WM_Anti,participantDirectory,'WM_Anti', grainity='rough')
plot_errorDistribution_relative(errors_dict_WM_Anti,participantDirectory,'WM_Anti', grainity='rough')

# WM Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distract', 'noResponse']
list2 = ['Circle', 'Nonagon', 'Heptagon', 'Pentagon', 'Triangle']
list3 = ['ClassYellow', 'ClassGreen', 'ClassBlue', 'ClassRed']
list4 = ['_simColor_simForm', '_simColor_diffForm', '_diffColor_simForm', '_diffColor_diffForm']
# Generating all combinations of categorical names
categorical_names_fineGrained = [''.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# info: DM error key pair - has to be added manually
list_error_keys = ['noResponse_simColor_simForm']

for j in list_error_keys:
    error_key_values = errors_dict_WM_Anti[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Anti')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'WM_Anti_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'WM_Anti_fineGrained ' + j, 'fine')

# WM Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx1 = {name: [] for name in categorical_names_WM_Ctx}
participantDirectory = directory + 'WM_Ctx1'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_WM_Ctx1 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx1, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_WM_Ctx1,participantDirectory,'WM_Ctx1', grainity='rough')
plot_errorDistribution_relative(errors_dict_WM_Ctx1,participantDirectory,'WM_Ctx1', grainity='rough')

# WM Ctx1 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['formClassCombi']
list2 = ['CircleCircle', 'PolygonPolygon', 'TriangleTriangle',\
         'CirclePolygon', 'CircleTriangle', 'PolygonTriangle']
list3 = ['colorClassCombi']
list4 = ['YellowYellow', 'GreenGreen', 'BlueBlue', 'RedRed', 'YellowGreen', 'YellowBlue', 'YellowRed',\
         'GreenBlue', 'GreenRed', 'BlueRed']
list5 = ['simColor_simForm', 'simColor_diffForm', 'diffColor_simForm', 'diffColor_diffForm']
list6 = ['responseMatch', 'responseMismatch', 'responseNoResponse']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5, list6)]
# info: DM error key pair - has to be added manually
list_error_keys = ['formClassCombi_CirclePolygon_simColor_simForm_responseMatch', 'formClassCombi_CirclePolygon_simColor_simForm_responseMatch']

for j in list_error_keys:
    error_key_values = errors_dict_WM_Ctx1[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Ctx1')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'WM_Ctx1_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'WM_Ctx1_fineGrained ' + j, 'fine')

# WM Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx2 = {name: [] for name in categorical_names_WM_Ctx}
participantDirectory = directory + 'WM_Ctx2'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
meta_files = glob.glob(os.path.join(participantDirectory, '*Meta.json'))
selected_npy_files = [file for file in npy_files if any(month in file for month in focusedMonths)]
selected_meta_files = [file for file in meta_files if any(month in file for month in focusedMonths)]

for npy_file, meta_file in zip(selected_npy_files, selected_meta_files):
    # Load the JSON content from the file
    with open(meta_file, 'r') as file:
        opened_meta_file = json.load(file)
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    errors_dict_WM_Ctx2 = get_errors_WM_Ctx(Response, errors_dict_WM_Ctx2, opened_meta_file)
# Visualize results
# plot_errorDistribution(errors_dict_WM_Ctx2,participantDirectory,'WM_Ctx2', grainity='rough')
plot_errorDistribution_relative(errors_dict_WM_Ctx2,participantDirectory,'WM_Ctx2', grainity='rough')

# WM Ctx2 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['formClassCombi']
list2 = ['CircleCircle', 'PolygonPolygon', 'TriangleTriangle',\
         'CirclePolygon', 'CircleTriangle', 'PolygonTriangle']
list3 = ['colorClassCombi']
list4 = ['YellowYellow', 'GreenGreen', 'BlueBlue', 'RedRed', 'YellowGreen', 'YellowBlue', 'YellowRed',\
         'GreenBlue', 'GreenRed', 'BlueRed']
list5 = ['simColor_simForm', 'simColor_diffForm', 'diffColor_simForm', 'diffColor_diffForm']
list6 = ['responseMatch', 'responseMismatch', 'responseNoResponse']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5, list6)]
# info: DM error key pair - has to be added manually
list_error_keys = ['formClassCombi_CirclePolygon_simColor_simForm_responseMatch']

for j in list_error_keys:
    error_key_values = errors_dict_WM_Ctx2[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Ctx2')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'WM_Ctx2_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, 'WM_Ctx2_fineGrained ' + j, 'fine')


