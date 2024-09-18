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
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Network import Model, get_perf
from Training import split_files
import Tools
import glob

########################################################################################################################
# Functions
########################################################################################################################
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
    sns.heatmap(data, annot=True, fmt='.0f', cmap='coolwarm', cbar=True,
                xticklabels=col_labels, yticklabels=row_labels, ax=ax,
                linewidths=0, linecolor='black', annot_kws={"fontsize": 22, "color": "white"}, square=True,
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
    subject = model_dir.split('\\')[-1].split('_')[2] + '_' + model_dir.split('\\')[-1].split('_')[3] # info: Subject of last model taken into account will define title
    plt.suptitle(f'{mode} Contingency Table (%) - {task} - {subject}', fontsize=18, y=.95, x=.4)

    # Adjust layout to fit all elements within the figure bounds
    plt.tight_layout(rect=[0, 0, .9, .95])  # Adjusted the layout to accommodate the title

    # Save the plot ensuring all elements are included

    save_path = os.path.join(figure_path, f'{mode}_Contingency_{task}_{subject}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight', pad_inches=0.2)

    # Display the plot
    plt.show()

def evaluate_model_responses(models, tasks):
    for task in tasks:
        accumulated_data_percentage = None
        model_count = 0
        # modelDirectories
        figurePath = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\Visuals\\Contingency'
        # Create directory for saving figures if it doesn't exist
        if not os.path.exists(figurePath):
            os.makedirs(figurePath)

        if len(models) > 1:
            mode = 'Average'
        else:
            mode = 'Individual'

        for model_dir in models:

            trial_dir = 'W:\\group_csp\\analyses\\oliver.frank\\Data' + '\\BeRNN_' + \
                        model_dir.split('\\')[-1].split('_')[3] + '\\PreprocessedData_wResp_ALL'

            # Info: Adjust for test and training data if training was random seeded
            file_triplets = get_file_triplets(trial_dir, task, model_dir)
            correct_match, error_match, correct_mismatch, error_mismatch, total_trials = evaluate_task(model_dir, task, file_triplets)

            # Calculate percentages for the contingency table
            data_percentage = np.round(
                np.array([[correct_match / total_trials * 100, correct_mismatch / total_trials * 100],
                          [error_match / total_trials * 100, error_mismatch / total_trials * 100]]), 2)

            # Accumulate the percentages
            if accumulated_data_percentage is None:
                accumulated_data_percentage = data_percentage
            else:
                accumulated_data_percentage += data_percentage

            model_count += 1

        # Average the accumulated percentages
        average_data_percentage = accumulated_data_percentage / model_count

        ratioCorrect = average_data_percentage[0][0] / average_data_percentage[0][1]
        ratioError = average_data_percentage[1][0] / average_data_percentage[1][1]

        # Visualize the averaged contingency table
        visualize_contingency_table(average_data_percentage, task, figurePath, mode, model_dir=model_dir, ratio_correct=ratioCorrect, ratio_error=ratioError)

def get_file_triplets(trial_dir, task, model_dir):
    dir = os.path.join(trial_dir, task)
    npy_files_Input = glob.glob(os.path.join(dir, '*Input.npy'))
    # Filter file_triplets for months taken into account
    monthsConsidered = list(range(int(model_dir.split('\\')[-1].split('_')[-1].split('-')[0]),
                                  int(model_dir.split('\\')[-1].split('_')[-1].split('-')[1]) + 1))
    for i in range(0, len(monthsConsidered)):
        monthsConsidered[i] = 'month_' + str(monthsConsidered[i])
    filtered_npy_files_Input = [file for file in npy_files_Input if any(month in file for month in monthsConsidered)]

    file_triplets = []
    for file in filtered_npy_files_Input:
        base_name = file.split('Input')[0]
        input_file = os.path.join(dir, base_name + 'Input.npy')
        yloc_file = os.path.join(dir, base_name + 'yLoc.npy')
        output_file = os.path.join(dir, base_name + 'Output.npy')
        file_triplets.append((input_file, yloc_file, output_file))

    # Split the file triplets and take the test files
    train_files, eval_files = split_files(file_triplets)

    return eval_files

def evaluate_task(model_dir, task, file_triplets):
    error_match = 0
    error_mismatch = 0
    correct_match = 0
    correct_mismatch = 0
    total_trials = 0
    # Prepare model restore
    hp = Tools.load_hp(model_dir)
    model = Model(model_dir, hp=hp)

    with tf.Session() as sess:
        model.restore(model_dir)

        for i in range(0, 1000): # Info: Each iteration represents one batch
            x, y, y_loc, base_name = Tools.load_trials(task, 'Evaluation', 40, file_triplets, True)
            ground_truth = np.load(os.path.join(base_name + 'Response.npy'), allow_pickle=True)

            # Sort response data
            c_mask = np.zeros((y.shape[0] * y.shape[1], y.shape[2]), dtype='float32')
            # Generate model response for the current batch
            feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp)
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

########################################################################################################################
# Model evaluation for chosen tasks
########################################################################################################################
# info: Create several tables for several models
models_list = [['W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_01_fR\\Model_2_BeRNN_01_Month_2-8'],
               ['W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_01_fR\\Model_3_BeRNN_01_Month_2-8',
                'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_01_fR\\Model_4_BeRNN_01_Month_2-8']]
# info: Tasks to evaluate
tasks = ['EF', 'WM']

for models in models_list:
    evaluate_model_responses(models, tasks)


