########################################################################################################################
# info: Error Comparison
########################################################################################################################
# Creates a contingency table that compares and classifies participant and model response as match (particpantResponse ==
# modelResponse) or mismatch (particpantResponse != modelResponse).
# The first word of the class represents the particpant's objective success in responding right or wrong to a trial and
# the second the model's success to reproduce this behavior.
########################################################################################################################

########################################################################################################################
# Strategies for errorComparison
########################################################################################################################

# Info: Error comparison on models trained with various data configurations
# The following sections compare error across different training and evaluation datasets.

# Info: Error comparison for models trained with complete data
# - Compare with complete data
# - Compare with error data
# - Compare with systematic error data
# - Compare with correct data
# - Compare with correct & systematic error data

# Info: Error comparison for models trained with correct & systematic error data
# - Compare with complete data
# - Compare with error data
# - Compare with systematic error data
# - Compare with correct data
# - Compare with correct & systematic error data

# Info: Error comparison for models trained with correct data
# - Compare with complete data
# - Compare with error data
# - Compare with systematic error data
# - Compare with correct data
# - Compare with correct & systematic error data

# Info: Error comparison for models trained with systematic error data
# - Compare with complete data
# - Compare with error data
# - Compare with systematic error data
# - Compare with correct data
# - Compare with correct & systematic error data

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # Ignore future warnings

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os

import Tools
from Network import Model, popvec, get_perf  # Import custom modules

########################################################################################################################
# Visualizes a contingency table with provided data and saves the plot.
########################################################################################################################
def visualize_contingencyTable(data, task, figurePath, model_dir, ratio_correct, ratio_error):
    # Define labels for the table
    row_labels = ['Participant Response Correct', 'Participant Response Incorrect']
    col_labels = ['Model Response Correct', 'Model Response Incorrect']

    # Plot the contingency table
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    cax = ax.matshow(data, cmap="coolwarm")  # Set colormap

    # Annotate cells with data values
    for (i, j), val in np.ndenumerate(data):
        ax.text(j, i, f'{val}', ha='center', va='center')

    # Set axis labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(col_labels, fontsize=12)
    ax.set_yticklabels(row_labels, fontsize=12)

    # Add color bar for visual reference
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_position([0.87, 0.275, 0.03, 0.45])  # Adjust position as needed

    # Display ratio text next to the table
    ax.text(2.2, 0, f'Ratio Correct: {ratio_correct:.2f}', va='center', fontsize=12, color='black')
    ax.text(2.2, 1, f'Ratio Error: {ratio_error:.2f}', va='center', fontsize=12, color='black')

    # Add title to the plot
    plt.title('Contingency table (%)' + ' - ' + task, fontsize=14)
    plt.subplots_adjust(left=0.4, right=0.85, top=0.9, bottom=0.1)  # Adjust layout

    # Save the plot to the specified directory
    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + task + '_CONTINGENCY.png'),
                format='png', dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()

########################################################################################################################
# Adjust paths as needed for your environment
########################################################################################################################
model = 'Model_129_BeRNN_05_Month_2-4'
model_dir = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\OLD' + '\\' + model  # Info: Adjust for non-HPT models
trial_dir = 'W:\\group_csp\\analyses\\oliver.frank\\Data' + '\\BeRNN_' + model.split('_')[
    3] + '\\PreprocessedData_wResp_ALL'  # Info: Adjust for test and training data
figurePath = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\Visuals\\Contingency\\' + model

# Create directory for saving figures if it doesn't exist
if not os.path.exists(figurePath):
    os.makedirs(figurePath)

########################################################################################################################
# Filter for months used in model training
########################################################################################################################
monthsConsidered = list(range(int(model.split('_')[-1].split('-')[0]), int(model.split('_')[-1].split('-')[1]) + 1))
for i in range(0, len(monthsConsidered)):
    monthsConsidered[i] = 'month_' + str(monthsConsidered[i])

########################################################################################################################
# Task iteration and data filtering
########################################################################################################################
tasks = ["DM", "DM_Anti", "EF", "EF_Anti", "RP", "RP_Anti", "RP_Ctx1", "RP_Ctx2", "WM", "WM_Anti", "WM_Ctx1",
         "WM_Ctx2"]  # List of tasks to evaluate

for task in tasks:
    # Filter files based on months considered
    npy_files_Input = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))

    # Exclude files with specific substrings in their names
    exclusion_list = ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']
    # Filter files: include only those that contain any of the strings in monthsConsidered
    filtered_npy_files_Input = [
        file for file in npy_files_Input
        if any(month in file for month in monthsConsidered) and not any(exclude in file for exclude in exclusion_list)
    ]

    ####################################################################################################################
    # Fix: Ideally, data should be split into the original train and test data used during model training - Not implemented so far.
    ####################################################################################################################

    # Collect all file triplets (Input, yLoc, Output) in the current subdirectory
    dir = os.path.join(trial_dir, task)
    file_triplets = []
    for file in os.listdir(dir):
        if file.endswith('Input.npy'):
            # Exclude files with specific substrings
            if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                continue
            base_name = file.split('Input')[0]
            input_file = os.path.join(dir, base_name + 'Input.npy')
            yloc_file = os.path.join(dir, base_name + 'yLoc.npy')
            output_file = os.path.join(dir, base_name + 'Output.npy')
            file_triplets.append((input_file, yloc_file, output_file))

    ####################################################################################################################
    # Load model and initialize count variables for error matching
    ####################################################################################################################
    hp = Tools.load_hp(model_dir)
    model = Model(model_dir, hp=hp)

    errorMatch = 0
    errorMisMatch = 0
    correctMatch = 0
    correctMisMatch = 0
    noCount = 0

    ####################################################################################################################
    # Evaluate model responses for each trial and compare with participant responses
    ####################################################################################################################
    with tf.Session() as sess:
        # Restore pre-trained model
        model.restore(model_dir)

        for i in range(0, 100):  # Info: Each iteration represents one batch
            mode = 'Evaluation'
            x, y, y_loc = Tools.load_trials(task, mode, 40, file_triplets,
                                            True)  # Load trials
            groundTruth = np.load(os.path.join(trial_dir, task, base_name + 'Response.npy'),
                                  allow_pickle=True)  # Load ground truth data

            # Sort response data
            c_mask = np.zeros((y.shape[0] * y.shape[1], y.shape[2]), dtype='float32')

            # Generate model response for the current batch
            feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp)
            c_lsq, c_reg, modelResponse_machineForm = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],
                                                               feed_dict=feed_dict)
            print('Model response for one batch created')

            # Compare model response with participant response
            perf = get_perf(modelResponse_machineForm, y_loc)
            print('Performance:', perf)

            # Classify responses based on participant correctness and model matching
            for i in range(0, len(perf)):
                if perf[i] == 1:  # Model response matches participant response
                    if groundTruth[0, i] == groundTruth[1, i]:  # Participant response is correct
                        correctMatch += 1
                    else:  # Participant response is incorrect
                        errorMatch += 1
                elif perf[i] == 0:  # Model response does not match participant response
                    if groundTruth[0, i] == groundTruth[1, i]:  # Participant response is correct
                        correctMisMatch += 1
                    else:  # Participant response is incorrect
                        errorMisMatch += 1
                else:
                    noCount += 1

    ####################################################################################################################
    # Calculate ratios and percentages for evaluation
    ####################################################################################################################
    totalAmountOfTrials = correctMatch + errorMatch + correctMisMatch + errorMisMatch

    ratioCorrect = correctMatch / correctMisMatch  # Ratio of correct model responses to incorrect model responses
    ratioError = errorMatch / errorMisMatch  # Ratio of correct model responses to incorrect model responses for participant errors

    percentageOfCorrectMatch = 100 / totalAmountOfTrials * correctMatch
    percentageOfErrorMatch = 100 / totalAmountOfTrials * errorMatch
    percentageOfCorrectMisMatch = 100 / totalAmountOfTrials * correctMisMatch
    percentageOfErrorMisMatch = 100 / totalAmountOfTrials * errorMisMatch

    totalPercentageOfMisMatch = percentageOfCorrectMatch + percentageOfErrorMisMatch

    ####################################################################################################################
    # Visualize the contingency table
    ####################################################################################################################
    data_percentage = np.round(np.array([[percentageOfCorrectMatch, percentageOfCorrectMisMatch],
                                         [percentageOfErrorMatch, percentageOfErrorMisMatch]]), 2)
    visualize_contingencyTable(data_percentage, task, figurePath, model_dir, ratioCorrect, ratioError)


