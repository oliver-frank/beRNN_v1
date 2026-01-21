# head: ################################################################################################################
# head: errorDistribution for each task individually
# head: ################################################################################################################
# Several functions to create a distribution of error classes for the particpant's behavior on the different task.
# The classes to be shown can be manually chosen by the user and can be rough- (few classes) or fine- (many classes)
# grained.

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import os
import json
import glob
import itertools
# import matplotlib.pyplot as plt
# import seaborn as sns

# attention: Fine Granularity doesn't work right at the moment - will be fixed soon
import os
import matplotlib.pyplot as plt
plt.ioff() # prevents windows to pop up when figs and plots are created
import seaborn as sns


def plot_errorDistribution_relative(errors_dict, directory, participant, task, granularity, titleAdd):
    # Prepare data for plotting
    categories = list(errors_dict.keys())
    total_occurrences = sum(len(values) for values in errors_dict.values())

    # Calculate relative occurrences (percentages)
    occurrences = [(len(values) / total_occurrences) * 100 for values in errors_dict.values()]

    # Filter out categories with no occurrences for labeling
    labels = [cat if len(errors_dict[cat]) > 0 else '' for cat in categories]

    # Set up the color palette
    palette = sns.color_palette("coolwarm", len(categories))

    # Create the bar chart
    # fig, ax = plt.subplots(figsize=(12, len(categories) * 0.5))  # Adjust figure size as needed
    fig, ax = plt.subplots(figsize=(4, 8))
    # sns.barplot(y=categories, x=occurrences, palette=palette, ax=ax)

    # Reverse the order for a nicer top-down view
    categories = categories[::-1]
    occurrences = occurrences[::-1]
    colors = palette[::-1]
    if granularity == 'rough':
        bar_thickness = 0.8  # Adjust this for finer/thicker bars (1.0 = full spacing)
    elif granularity == 'fine':
        bar_thickness = 20.0

    # Draw horizontal bars manually
    for i, (cat, val, color) in enumerate(zip(categories, occurrences, colors)):
        ax.barh(i, val, color=color, height=bar_thickness)

    # Set y-ticks to match the categories
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(labels[::-1], fontsize=8)

    # Set labels and titles
    ax.set_xlabel('Relative Occurrences (%)', fontsize=10, labelpad=10)
    ax.tick_params(axis='x', labelsize=8)
    ax.set_title(f'{participant} {task} {titleAdd}', fontsize=12, pad=20)

    # # Keep the tick labels on the left
    # ax.set_yticks(range(len(categories)))
    # ax.set_yticklabels(labels, fontsize=12)

    # Add gridlines for better readability
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)

    # Set x-axis limits based on granularity
    if granularity == 'rough':
        ax.set_xlim([0, 100])
    elif granularity == 'fine':
        ax.set_xlim([0, 100])

    # Move the y-axis label to the right side
    ax_right = ax.twinx()  # Create a twin Axes sharing the x-axis
    ax_right.set_ylabel('Error Categories', fontsize=10, labelpad=10)
    ax_right.yaxis.set_label_position("right")  # Move the label to the right
    ax_right.yaxis.tick_right()  # Ensure ticks are not visible on the right

    # Hide the tick marks and labels on the right y-axis
    ax_right.set_yticks([])  # No ticks on the right
    ax_right.set_yticklabels([])  # No labels on the right

    # Improve layout and adjust spacing
    plt.tight_layout()
    plt.subplots_adjust(left=0.3, right=0.95, bottom=0.1, top=0.9)

    # Save plot with bbox_inches='tight' to prevent label cutoff
    save_path = os.path.join(directory.split('\\data')[0], 'ErrorGraphics', task, f'errorDistribution_{participant}_{task}_{titleAdd}_relative.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=100, bbox_inches='tight')

    # Show plot
    # plt.show()

    # info: For fine granularity return the names of the categories with highest occurrences
    if granularity == 'rough':
        # Count occurrences per category
        category_counts = {cat: len(values) for cat, values in errors_dict.items()}
        # Sort categories by count in descending order
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        # Extract top 3 category names
        top_four_categories = [cat for cat, count in sorted_categories[:4]]

        return top_four_categories

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
            if sortedResponse[0, i] == '000_000.png' or sortedResponse[0, i] == 'BEGIN' or sortedResponse[0, i] == 'nan' or sortedResponse[0, i] == None: # errors where the wrong stim was hitten
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

            # errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1] # info: Don't count 3stim extra as there would be too many error classes

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

            # Missed correct stimulus
            correctColorClass = next(
                (cls for cls, colors in colorDict.items() if sortedResponse[2, i].split('_')[0] in colors), None)
            distractColorClass = next(
                (cls for cls, colors in colorDict.items() if sortedResponse[3, i].split('_')[0] in colors), None)
            if correctColorClass == distractColorClass:
                errorComponent_3 = 'simColor'
            else:
                errorComponent_3 = 'diffColor'

            correctFormClass = next((cls for cls, forms in formDict.items() if
                                     sortedResponse[2, i].split('_')[1].split('.')[0] in forms), None)
            distractFormClass = next((cls for cls, forms in formDict.items() if
                                      sortedResponse[3, i].split('_')[1].split('.')[0] in forms), None)
            if correctFormClass == distractFormClass:
                errorComponent_4 = 'simForm'
            else:
                errorComponent_4 = 'diffForm'

            # errorComponent_3 = opened_meta_file['difficultyLevel'].split('trials_')[1] # info: Don't count 3stim extra as there would be too many error classes
            if sortedResponse[0, i] == 'noResponse':
                errorComponent_5 = 'response' + 'NoResponse'
            else:
                errorComponent_5 = 'response' + sortedResponse[0, i]

            # Concatenate error components
            currentChosenList = f'{errorComponent_1}_{errorComponent_2}_{errorComponent_3}_{errorComponent_4}_{errorComponent_5}'
            errors_dict_fineGrained[currentChosenList].append(sortedResponse[:, i])
    return errors_dict_fineGrained

participant = 'beRNN_03'
# focusedMonths = ['month_1','month_2','month_3','month_4','month_5','month_6','month_7','month_8','month_9','month_10','month_11','month_12' ]
focusedMonths = ['month_4','month_5','month_6']
directory = f'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\{participant}\\data_highDim' # info: whole script made for original dataset only


# info: ################################################################################################################
# info: Decision Making
# info: ################################################################################################################
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

########################################################################################################################
# DM -------------------------------------------------------------------------------------------------------------------
errors_dict_DM = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'DM')
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that HIGHER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_descending(Response)
    errors_dict_DM = get_errors_DM(sortedResponse, errors_dict_DM, distract_dict, opposite_dict, strength_dict)
# Visualize results
top_four_categories = plot_errorDistribution_relative(errors_dict_DM, participantDirectory, participant, 'DM', 'rough', titleAdd = 'all')

# DM - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Lowest', 'Low', 'Strong', 'Strongest']
list3 = ['correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Lowest', 'Low', 'Strong', 'Strongest']
list5 = ['responseNoResponse', 'responseNan', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_DM[j]
    sortedResponse = sort_rows_descending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'DM')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'DM', 'fine', titleAdd = j)

########################################################################################################################
# DM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_DM_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'DM_Anti')
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_DM_Anti = get_errors_DM(sortedResponse, errors_dict_DM_Anti, distract_dict, opposite_dict, strength_dict)
# Visualize results
top_four_categories = plot_errorDistribution_relative(errors_dict_DM_Anti,participantDirectory, participant, 'DM_Anti', 'rough', titleAdd = 'all')

# DM Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Lowest', 'Low', 'Strong', 'Strongest']
list3 = ['correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Lowest', 'Low', 'Strong', 'Strongest']
list5 = ['responseNoResponse', 'responseNoresponse', 'responseNan', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_DM_Anti[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'DM_Anti')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'DM_Anti', 'fine', titleAdd=j)


# info: ################################################################################################################
# info: Executive Functioning
# info: ################################################################################################################
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

########################################################################################################################
# EF -------------------------------------------------------------------------------------------------------------------
errors_dict_EF = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'EF')
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF = get_errors_EF(sortedResponse, errors_dict_EF, distract_dict, opposite_dict)
# Visualize results
top_four_categories = plot_errorDistribution_relative(errors_dict_EF, participantDirectory, participant, 'EF', 'rough', titleAdd = 'all')

# EF - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractX', 'distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Green', 'Red']
list3 = ['noResponse', 'correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Green', 'Red']
list5 = ['responsenoResponse', 'responsenan', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_EF[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'EF')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'EF', 'fine', titleAdd=j)

########################################################################################################################
# EF Anti --------------------------------------------------------------------------------------------------------------
errors_dict_EF_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'EF_Anti')
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
selected_months_files = [file for file in npy_files if any(month in file for month in focusedMonths)]

for npy_file in selected_months_files:
    # Use the function
    Response = np.load(npy_file, allow_pickle=True)
    # Sort the 4th and 5th row, so that LOWER value is on 4th row. Sort 2nd and 3rd accordingly
    sortedResponse = sort_rows_ascending(Response)
    errors_dict_EF_Anti = get_errors_EF(Response, errors_dict_EF_Anti, distract_dict, opposite_dict)
# Visualize results
top_four_categories = plot_errorDistribution_relative(errors_dict_EF_Anti, participantDirectory, participant, 'EF_Anti', 'rough', titleAdd = 'all')

# EF Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractX', 'distractLeft', 'distractRight', 'distractUp', 'distractDown']
list2 = ['Green', 'Red']
list3 = ['noResponse', 'correctLeft', 'correctRight', 'correctUp', 'correctDown']
list4 = ['Green', 'Red']
list5 = ['responsenoResponse', 'responsenan', 'responseL', 'responseR', 'responseU', 'responseD']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4, list5)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_EF_Anti[j]
    sortedResponse = sort_rows_ascending(np.column_stack(error_key_values))
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'EF_Anti')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'EF_Anti', 'fine', titleAdd=j)


# info: ################################################################################################################
# info: Relational Processing
# info: ################################################################################################################
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

########################################################################################################################
# RP -------------------------------------------------------------------------------------------------------------------
errors_dict_RP = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'RP')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_RP,participantDirectory, participant, 'RP', 'rough', titleAdd = 'all')

# RP - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_RP[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'RP', 'fine', titleAdd=j)

########################################################################################################################
# RP Anti --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Anti = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'RP_Anti')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Anti,participantDirectory, participant, 'RP_Anti', 'rough', titleAdd = 'all')

# RP Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_RP_Anti[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Anti')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'RP_Anti', 'fine', titleAdd=j)

########################################################################################################################
# RP Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx1 = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'RP_Ctx1')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Ctx1,participantDirectory, participant, 'RP_Ctx1', 'rough', titleAdd = 'all')

# RP Ctx1 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_RP_Ctx1[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Ctx1')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'RP_Ctx1', 'fine', titleAdd=j)

########################################################################################################################
# RP Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_RP_Ctx2 = {name: [] for name in categorical_names}
# Get list of necessary files in directory
participantDirectory = os.path.join(directory, 'RP_Ctx2')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_RP_Ctx2,participantDirectory, participant, 'RP_Ctx2', 'rough', titleAdd = 'all')

# RP Ctx2 - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distractCircle', 'distractNonagon', 'distractHeptagon', 'distractPentagon', 'distractTriangle', 'distractNoResponse']
list2 = ['Amber', 'Blue', 'Green', 'Lime', 'Magenta', 'Moss', 'Orange', 'Purple', 'Red', 'Rust', 'Violet', 'Yellow']
list3 = ['correctCircle', 'correctNonagon', 'correctHeptagon', 'correctPentagon', 'correctTriangle']
list4 = ['Similiar', 'NonSimiliar']
# Generating all combinations of categorical names
categorical_names_fineGrained = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_RP_Ctx2[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'RP_Ctx2')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'RP_Ctx2', 'fine', titleAdd=j)


# info: ################################################################################################################
# info: Working Memory
# info: ################################################################################################################
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
list2 = ['3stim']
list3 = ['diffColor_diffForm', 'simColor_diffForm', 'simColor_simForm']

categorical_names_WM_1 = ['_'.join(combination) for combination in itertools.product(list1, list2, list3)]
categorical_names_WM_2 = ['_'.join(combination) for combination in itertools.product(list1, list3)]

categorical_names_WM = categorical_names_WM_1 + categorical_names_WM_2

# Create categorical names for WM_Ctx tasks
list1 = ['formClassCombi_CircleCircle', 'formClassCombi_CirclePolygon', 'formClassCombi_CircleTriangle',\
         'formClassCombi_PolygonPolygon', 'formClassCombi_PolygonTriangle', 'formClassCombi_TriangleTriangle']
list2 = ['3stim'] # info: naming for ctx tasks accidentally different than above
list3 = ['diffColor_diffForm', 'simColor_diffForm','diffColor_simForm', 'simColor_simForm']
list4 = ['responseMatch', 'responseMismatch', 'responseNoResponse']

categorical_names_WM_Ctx_1 = ['_'.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
categorical_names_WM_Ctx_2 = ['_'.join(combination) for combination in itertools.product(list1, list3, list4)]

categorical_names_WM_Ctx = categorical_names_WM_Ctx_2 + categorical_names_WM_Ctx_1

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
                print('Error occured: ', e) # info: Happens rarely if accidentally one of two stims is None
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

########################################################################################################################
# WM -------------------------------------------------------------------------------------------------------------------
errors_dict_WM = {name: [] for name in categorical_names_WM}
participantDirectory = os.path.join(directory, 'WM')
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
# Visualize results
top_four_categories = plot_errorDistribution_relative(errors_dict_WM,participantDirectory, participant, 'WM', 'rough', titleAdd = 'all')

# WM - Fine Graining ---------------------------------------------------------------------------------------------------
list1 = ['distract', 'noResponse']
list2 = ['Circle', 'Nonagon', 'Heptagon', 'Pentagon', 'Triangle']
list3 = ['ClassYellow', 'ClassGreen', 'ClassBlue', 'ClassRed']
list4 = ['_simColor_simForm', '_simColor_diffForm', '_diffColor_simForm', '_diffColor_diffForm']
# Generating all combinations of categorical names
categorical_names_fineGrained = [''.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_WM[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'WM', 'fine', titleAdd=j)

########################################################################################################################
# WM Anti --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Anti = {name: [] for name in categorical_names_WM}
participantDirectory = os.path.join(directory, 'WM_Anti')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Anti,participantDirectory, participant, 'WM_Anti', 'rough', titleAdd = 'all')

# WM Anti - Fine Graining ----------------------------------------------------------------------------------------------
list1 = ['distract', 'noResponse']
list2 = ['Circle', 'Nonagon', 'Heptagon', 'Pentagon', 'Triangle']
list3 = ['ClassYellow', 'ClassGreen', 'ClassBlue', 'ClassRed']
list4 = ['_simColor_simForm', '_simColor_diffForm', '_diffColor_simForm', '_diffColor_diffForm']
# Generating all combinations of categorical names
categorical_names_fineGrained = [''.join(combination) for combination in itertools.product(list1, list2, list3, list4)]
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_WM_Anti[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Anti')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'WM_Anti', 'fine', titleAdd=j)

#######################################################################################################################
# WM Ctx1 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx1 = {name: [] for name in categorical_names_WM_Ctx}
participantDirectory = os.path.join(directory, 'WM_Ctx1')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Ctx1,participantDirectory, participant, 'WM_Ctx1', 'rough', titleAdd = 'all')

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
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_WM_Ctx1[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Ctx1')
    # plot_errorDistribution(errors_dict_fineGrained, participantDirectory, 'WM_Ctx1_fineGrained ' + j, 'fine')
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'WM_Ctx1', 'fine', titleAdd=j)

########################################################################################################################
# WM Ctx2 --------------------------------------------------------------------------------------------------------------
errors_dict_WM_Ctx2 = {name: [] for name in categorical_names_WM_Ctx}
participantDirectory = os.path.join(directory, 'WM_Ctx2')
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
top_four_categories = plot_errorDistribution_relative(errors_dict_WM_Ctx2,participantDirectory, participant, 'WM_Ctx2', 'rough', titleAdd = 'all')

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
# Define the four error classes to plot in fine granularity
list_error_keys = [top_four_categories[0], top_four_categories[1], top_four_categories[2], top_four_categories[3]]

for j in list_error_keys:
    error_key_values = errors_dict_WM_Ctx2[j]
    sortedResponse = np.column_stack(error_key_values)
    # Creating dict with created names
    errors_dict_fineGrained = {name: [] for name in categorical_names_fineGrained}
    errors_dict_fineGrained = get_fine_grained_error(sortedResponse, errors_dict_fineGrained, 'WM_Ctx2')
    # Visualize results
    plot_errorDistribution_relative(errors_dict_fineGrained, participantDirectory, participant, 'WM_Ctx2', 'fine',titleAdd=j)



# head: ################################################################################################################
# head: contingencyTable for each task individually
# head: ################################################################################################################
# Creates a contingency table that compares and classifies participant and model response as match (particpantResponse ==
# modelResponse) or mismatch (particpantResponse != modelResponse).
# The first word of the class represents the particpant's objective success in responding right or wrong to a trial and
# the second the model's success to reproduce this behavior.
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from network import Model, get_perf
from tools import split_files
import tools
import glob

########################################################################################################################
# Functions
########################################################################################################################
def evaluate_model_responses(directory, dataDirectory, models, tasks):
    accumulated_data_percentage = 0
    task_count = 0
    for i, task in enumerate(tasks):
        # modelDirectories
        figurePath = os.path.join(directory, 'contingencyTable')
        # Create directory for saving figures if it doesn't exist
        if not os.path.exists(figurePath):
            os.makedirs(figurePath)

        # mode = 'Average' for the future
        mode = 'Average'

        for model_dir in models:
            correct_match, error_match, correct_mismatch, error_mismatch, total_trials = evaluate_task(model_dir, task, i)

            # Calculate percentages for the contingency table
            data_percentage = np.round(
                np.array([[correct_match / total_trials * 100, correct_mismatch / total_trials * 100],
                          [error_match / total_trials * 100, error_mismatch / total_trials * 100]]), 2)
            # data_percentage = np.round(
            #     np.array([[(correct_match / total_trials * 100) - 5, (correct_mismatch / total_trials * 100) + 5],
            #               [(error_match / total_trials * 100) - 5, (error_mismatch / total_trials * 100) + 5]]), 2)

            # Accumulate the percentages
            if accumulated_data_percentage is None:
                accumulated_data_percentage = data_percentage
            else:
                accumulated_data_percentage += data_percentage

        task_count += 1

    # Average the accumulated percentages
    average_data_percentage = accumulated_data_percentage / task_count

    ratioCorrect = average_data_percentage[0][0] / average_data_percentage[0][1]
    ratioError = average_data_percentage[1][0] / average_data_percentage[1][1]

    # Visualize the averaged contingency table
    visualize_contingency_table(average_data_percentage, task, figurePath, mode, model_dir=model_dir, ratio_correct=ratioCorrect, ratio_error=ratioError)

def createSplittedDatasets(hp, preprocessedData_path, month):
    # Split the data into training and test data -----------------------------------------------------------------------
    # List of the subdirectories
    subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if
               os.path.isdir(os.path.join(preprocessedData_path, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                if month not in file:  # Sort out months which should not be considered
                    continue
                # Add all necessary files to triplets
                base_name = file.split('Input')[0]
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                response_file = os.path.join(subdir, base_name + 'Response.npy')

                file_quartett.append((input_file, yloc_file, output_file, response_file))

        # Split the file triplets
        train_files, eval_files = split_files(hp, file_quartett)

        # Store the results in the dictionaries
        train_data[subdir] = train_files
        eval_data[subdir] = eval_files

    return train_data, eval_data

def evaluate_task(model_dir, task, indice):
    error_match = 0
    error_mismatch = 0
    correct_match = 0
    correct_mismatch = 0
    total_trials = 0

    # Prepare model restore
    hp = tools.load_hp(model_dir)

    hp['rng'] = np.random.default_rng(42) # as for training

    file_quartett = []

    for month in hp['monthsConsidered']:
        train_data, eval_data = createSplittedDatasets(hp, dataDirectory, month)
        file_quartett_ = eval_data[list(eval_data)[indice]]

        file_quartett.extend(file_quartett_)

    hp['rng'] = np.random.default_rng() # as for training
    model = Model(model_dir, hp=hp)

    with tf.Session() as sess:
        model.restore(model_dir)

        for i in range(0, 10): # Info: Each iteration represents one batch - after the spliting most often you only have 4 batches for one month

            x, y, y_loc, base_name = tools.load_trials(hp['rng'], task, 'test', 40, file_quartett, True)
            ground_truth = np.load(os.path.join(base_name + 'Response.npy'), allow_pickle=True)

            print(base_name)

            # Sort response data
            c_mask = np.zeros((y.shape[0] * y.shape[1], y.shape[2]), dtype='float32')
            # Generate model response for the current batch
            feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)
            c_lsq, c_reg, modelResponse_machineForm = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],feed_dict=feed_dict)

            perf = get_perf(modelResponse_machineForm, y_loc)

            for i in range(0,len(perf)):
                if perf[i] == 1:  # Model response matches participant response
                    if ground_truth[0, i] == ground_truth[1, i]:
                        correct_match += 1
                    else:
                        error_match += 1
                elif perf[i] == 0:  # Model response does not match participant response
                    if ground_truth[0, i] == ground_truth[1, i]:
                        correct_mismatch += 1
                    else:
                        error_mismatch += 1
            total_trials += len(perf)

    return correct_match, error_match, correct_mismatch, error_mismatch, total_trials

def visualize_contingency_table(data, task, figure_path, mode, model_dir, ratio_correct, ratio_error):
    # Define labels for the table
    row_labels = ['Participant Response Correct', 'Participant Response Incorrect']
    col_labels = ['Model Response Correct', 'Model Response Incorrect']

    # Increase the figure size for better alignment and spacing
    fig, ax = plt.subplots(figsize=(12, 10))  # Increased size for the table

    # Create a divider to manage the color bar size relative to the heatmap
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)

    # Plot the contingency table with clear annotations
    sns.heatmap(data, annot=True, fmt='.0f', cmap='Greys', cbar=True,
                xticklabels=col_labels, yticklabels=row_labels, ax=ax,
                linewidths=0, linecolor='black', annot_kws={"fontsize": 30, "color": "black"}, square=True,
                cbar_ax=cax,
                vmin=0, vmax=100)

    # Set axis labels with adequate spacing and no rotation
    ax.set_xticklabels(col_labels, fontsize=16)
    ax.set_yticklabels(row_labels, fontsize=16, rotation=90)

    # Configure the color bar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    # Add ratio text to the right of the heatmap, making it closer but avoiding overlap
    ax.text(2.3, 0.5, f'Ratio Correct: {ratio_correct:.2f}', va='center', fontsize=18, color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    ax.text(2.3, 1.5, f'Ratio Error: {ratio_error:.2f}', va='center', fontsize=18, color='black',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    # Add title directly above the heatmap in the center
    subject = '_'.join(model_dir.split('\\')[-2].split('_')[0:2])
    plt.suptitle(f'{subject}', fontsize=18, y=.95, x=.4)

    # Adjust layout to fit all elements within the figure bounds
    plt.tight_layout(rect=[0, 0, .9, .95])  # Adjusted the layout to accommodate the title

    # Save the plot ensuring all elements are included

    save_path = os.path.join(figure_path, f'{mode}_Contingency_{subject}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)

    # Display the plot
    # plt.show()

########################################################################################################################
# Model evaluation for chosen tasks
########################################################################################################################
# Directory has to point exactly on the model folder of interest
participant = 'beRNN_02'
directory = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_eR_multiTask_{participant}_data_highDim_hp_2\highDim\{participant}\2\{participant}_AllTask_4-6_highDim_iter21_LeakyRNN_diag_256_relu'
dataDirectory = f'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data\\{participant}\\data_highDim' # add right participant and dataset
modelsList = [os.path.join(directory, item) for item in os.listdir(directory) if 'model_month_6' in item]

# Tasks to evaluate
tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
evaluate_model_responses(directory, dataDirectory, modelsList, tasks)


