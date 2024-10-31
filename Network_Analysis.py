########################################################################################################################
# info: Network Analysis
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('WebAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'wxAgg'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from analysis import clustering #, variance
import Tools
from Tools import rule_name
from Network import Model
import tensorflow as tf


selected_hp_keys = ['rnn_type', 'activation', 'tau', 'dt', 'sigma_rec', 'sigma_x', 'w_rec_init', 'l1_h', 'l2_h', \
                    'l1_weight', 'l2_weight', 'l2_weight_init', 'learning_rate', 'n_rnn', 'c_mask_responseValue',
                    'monthsConsidered']  # Replace with the keys you want



########################################################################################################################
# Pre-Allocation of variables and models to be examined (+ definition of one global function)
########################################################################################################################
folder = '\\barnaModels\\cascade2\\beRNN_03'
folderPath = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\barnaModels\\Cascade2\\beRNN_03'
# folderPath = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels' + folder
figurePath = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\visuals'

# ATTENTION: You have to save the .npy files somewhere
figurePath_performance = figurePath + '\\performance' + folder
figurePath_functionalCorrelation = figurePath + '\\functionalCorrelation' + folder
figurePath_structuralCorrelation = figurePath + '\\structuralCorrelation' + folder
figurePath_taskVariance = figurePath + '\\taskVariance' + folder

# figurePath_performance = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\visuals\\performance' + folder
# figurePath_functionalCorrelation = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\visuals\\functionalCorrelation' + folder
# figurePath_structuralCorrelation = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\visuals\\structuralCorrelation' + folder
# figurePath_taskVariance = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\visuals\\taskVariance' + folder

figurePaths = [figurePath_performance, figurePath_functionalCorrelation, figurePath_structuralCorrelation, figurePath_taskVariance]

for figurePath in figurePaths:
    # Check if the directory exists
    if not os.path.exists(figurePath):
        # If it doesn't exist, create the directory
        os.makedirs(figurePath)
        print(f"Directory created: {figurePath}")
    else:
        print(f"Directory already exists: {figurePath}")

files = os.listdir(folderPath)
model_list = []
for file in files:
    # if any(include in file for include in ['Model_1_', 'Model_6_']):
    model_list.append(os.path.join(folderPath,file))

def smoothed(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    # Calculate how many points we need to add to match the length of the original data
    padding_length = len(data) - len(smoothed_data)
    if padding_length > 0:
        last_value = smoothed_data[-1]
        smoothed_data = np.concatenate((smoothed_data, [last_value] * padding_length))
    return smoothed_data



########################################################################################################################
# Performance - Individual network
########################################################################################################################
# Note to visualization of training and test performance: The test data gives for maxsteps of 1e7 5000 performance data
# points, each representing 800 evaluated trials. The training data gives for maxsteps of 1e7 25000 performance data points,
# each representing 40 trained trials. So I should gather 5 data points of the training data to have the same smoothness
# in the plots, window size = 5
########################################################################################################################
def plot_performanceprogress_eval_BeRNN(model_dir, figurePath, rule_plot=None):
    # Plot Evaluation Progress
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
    trials = log['trials']
    x_plot = np.array(trials) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_eval = plt.figure(figsize=(14, 6))
    ax = fig_eval.add_axes([0.1, 0.4, 0.6, 0.5])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    if rule_plot == None:
        rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        line = ax.plot(x_plot, np.log10(log['cost_' + rule]), color=rule_color[rule])
        ax.plot(x_plot, log['perf_' + rule], color=rule_color[rule])
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 80000])
    ax.set_xlabel('Total number of trials (/1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add Hyperparameters as Text Box
    hp_text = "\n".join([f"{key}: {currentHP[key]}" for key in selected_hp_keys if key in currentHP])
    plt.figtext(0.75, 0.75, hp_text, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

    lg = fig_eval.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.1),
                         # co: first value influences horizontal position of legend
                         fontsize=fs, labelspacing=0.3, loc=6, frameon=False)
    plt.setp(lg.get_title(), fontsize=fs)
    # plt.title(model_dir.split("\\")[-1]+'_EVALUATION.png') # info: Add title
    plt.title('_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST', fontsize=18)  # info: Add title
    # # Add the randomness thresholds
    # # DM & RP Ctx
    # plt.axhline(y=0.2, color='green', label= 'DM & DM Anti & RP Ctx1 & RP Ctx2', linestyle=':')
    # plt.axhline(y=0.2, color='green', linestyle=':')
    # # # EF
    # plt.axhline(y=0.25, color='black', label= 'EF & EF Anti', linestyle=':')
    # plt.axhline(y=0.25, color='black', linestyle=':')
    # # # RP
    # plt.axhline(y=0.143, color='brown', label= 'RP & RP Anti', linestyle=':')
    # plt.axhline(y=0.143, color='brown', linestyle=':')
    # # # WM
    # plt.axhline(y=0.5, color='blue', label= 'WM & WM Anti & WM Ctx1 & WM Ctx2', linestyle=':')
    # plt.axhline(y=0.5, color='blue', linestyle=':')
    # #
    # rt = fig.legend(title='Randomness threshold', bbox_to_anchor=(0.1, 0.35), fontsize=fs, labelspacing=0.3  # co: first value influences length of
    #                 ,loc=6, frameon=False)
    # plt.setp(rt.get_title(), fontsize=fs)

    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_TEST.png'), format='png', dpi=300)

    plt.show()

def plot_performanceprogress_train_BeRNN(model_dir, figurePath, rule_plot=None):
    # Plot Training Progress
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
    trials = log['trials']  # info: There is an entry every 40 trials for each task
    x_plot = (np.array(trials)) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_train = plt.figure(figsize=(14, 6))
    ax = fig_train.add_axes([0.1, 0.4, 0.6, 0.5])  # info: third value influences width of cartoon
    lines = list()
    labels = list()

    if rule_plot is None:
        rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        y_cost = log['cost_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]
        y_perf = log['perf_train_' + rule][::int((len(log['perf_train_' + rule]) / len(x_plot)))][:len(x_plot)]

        window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation

        y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        y_perf_smoothed = smoothed(y_perf, window_size=window_size)

        # Ensure the lengths match
        y_cost_smoothed = y_cost_smoothed[:len(x_plot)]
        y_perf_smoothed = y_perf_smoothed[:len(x_plot)]

        line = ax.plot(x_plot, np.log10(y_cost_smoothed), color=rule_color[rule])
        ax.plot(x_plot, y_perf_smoothed, color=rule_color[rule])

        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (/1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Add Hyperparameters as Text Box
    hp_text = "\n".join([f"{key}: {currentHP[key]}" for key in selected_hp_keys if key in currentHP])
    plt.figtext(0.75, 0.75, hp_text, fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', alpha=0.5))

    lg = fig_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.1),
                          fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    plt.setp(lg.get_title(), fontsize=fs)

    plt.title('_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING', fontsize=18)

    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_TRAINING.png'), format='png', dpi=300)

    plt.show()



# # ########################################################################################################################
# # # Performance - Group of networks
# # ########################################################################################################################
# def aggregate_performance_eval_data(model_list, tasks):
#     aggregated_costs = {task: [] for task in tasks}
#     aggregated_performances = {task: [] for task in tasks}
#
#     for model_dir in model_list:
#         log = Tools.load_log(model_dir)
#         hp = Tools.load_hp(model_dir)
#
#         trials = log['trials']
#         x_plot = (np.array(trials)) / 1000  # scale the x-axis right
#
#         for task in tasks:
#             y_cost = log['cost_' + task]
#             y_perf = log['perf_' + task]
#
#             aggregated_costs[task].append(y_cost)
#             aggregated_performances[task].append(y_perf)
#
#     return aggregated_costs, aggregated_performances, x_plot
#
# def aggregate_performance_train_data(model_list, tasks):
#     aggregated_costs = {task: [] for task in tasks}
#     aggregated_performances = {task: [] for task in tasks}
#
#     for model_dir in model_list:
#         log = Tools.load_log(model_dir)
#         hp = Tools.load_hp(model_dir)
#
#         trials = log['trials']
#         x_plot = (np.array(trials)) / 1000  # scale the x-axis right
#
#
#         for task in tasks:
#             y_cost = log['cost_train_' + task][::int((len(log['cost_train_' + task]) / len(x_plot)))][:len(x_plot)]
#             y_perf = log['perf_train_' + task][::int((len(log['cost_train_' + task]) / len(x_plot)))][:len(x_plot)]
#
#             window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation
#
#             y_cost_smoothed = smoothed(y_cost, window_size=window_size)
#             y_perf_smoothed = smoothed(y_perf, window_size=window_size)
#
#             # Ensure the lengths match
#             y_cost_smoothed = y_cost_smoothed[:len(x_plot)]
#             y_perf_smoothed = y_perf_smoothed[:len(x_plot)]
#
#             aggregated_costs[task].append(y_cost_smoothed)
#             aggregated_performances[task].append(y_perf_smoothed)
#
#     return aggregated_costs, aggregated_performances, x_plot
#
# def plot_aggregated_performance(model_list, mode, tasks, figure_path):
#     """
#     Plots the aggregated performance of models in training or evaluation mode across multiple tasks.
#     Parameters:
#         model_list (list): List of models to aggregate performance from.
#         mode (str): Mode of operation, either 'train' or 'eval'.
#         tasks (list): List of tasks for which to aggregate performance.
#         figure_path (str): Path to save the resulting figure.
#     """
#
#     # Select the correct aggregation function based on mode
#     if mode == 'train':
#         aggregated_costs, aggregated_performances, x_plot = aggregate_performance_train_data(model_list, tasks)
#         modus = 'Training'
#     elif mode == 'eval':
#         aggregated_costs, aggregated_performances, x_plot = aggregate_performance_eval_data(model_list, tasks)
#         modus = 'Evaluation'
#
#     # Create the plot
#     fs = 18  # Set font size to match the second function
#     fig, ax = plt.subplots(figsize=(14, 6))
#     lines = []
#     labels = []
#
#     for task in tasks:
#         # Convert list of arrays to a 2D array for easier mean/std calculation
#         costs_array = np.array(aggregated_costs[task])
#         performances_array = np.array(aggregated_performances[task])
#
#         mean_costs = np.mean(costs_array, axis=0)
#         std_costs = np.std(costs_array, axis=0)
#         mean_performances = np.mean(performances_array, axis=0)
#         std_performances = np.std(performances_array, axis=0)
#
#         # Plot performance
#         line, = ax.plot(x_plot, mean_performances, color=rule_color[task], linestyle='-', label=task)
#         ax.fill_between(x_plot, mean_performances - std_performances, mean_performances + std_performances,
#                         color=rule_color[task], alpha=0.1)
#
#         lines.append(line)
#         labels.append(rule_name[task])
#
#     # Set labels and axis settings
#     ax.tick_params(axis='both', which='major', labelsize=fs)
#     ax.set_ylim([0, 1])
#     ax.set_xlabel('Total number of trials (/1000)', fontsize=fs, labelpad=2)
#     ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
#     ax.locator_params(axis='x', nbins=5)
#     ax.set_yticks([0, .25, .5, .75, 1])
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#
#     # Adjust the subplot to make space for the legend below
#     fig.subplots_adjust(bottom=0.3)  # Increased from 0.25 to 0.3 to create more space
#
#     # Place the legend in a similar style to the second function but adjust its position slightly
#     lg = fig.legend(lines, labels, title='Tasks', ncol=2, bbox_to_anchor=(0.5, -0.25),
#                     fontsize=fs, labelspacing=0.3, loc='upper center', frameon=False)
#     plt.setp(lg.get_title(), fontsize=fs)
#
#     # Title
#     subject = '_'.join(model_list[0].split("\\")[-1].split('_')[2:4])
#     plt.title(f'Average {modus} Performance Across Networks - {subject}', fontsize=fs)
#
#     # Save the figure
#     model_name = '_'.join(model_list[0].split("\\")[-1].split('_')[1:6])
#     plt.savefig(os.path.join(figure_path, f'modelAverage_{model_name}_{modus}.png'), format='png', dpi=300)
#
#     plt.show()
#
# # # Assign a color to each task
# # _rule_color = {
# #                 'DM': 'green',
# #                 'DM_Anti': 'olive',
# #                 'EF': 'forest green',
# #                 'EF_Anti': 'mustard',
# #                 'RP': 'tan',
# #                 'RP_Anti': 'brown',
# #                 'RP_Ctx1': 'lavender',
# #                 'RP_Ctx2': 'aqua',
# #                 'WM': 'bright purple',
# #                 'WM_Anti': 'green blue',
# #                 'WM_Ctx1': 'blue',
# #                 'WM_Ctx2': 'indigo'
# #                 }
# # rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}
# # # Define all tasks involved
# # tasks = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2', 'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']
# #
# # # Plot the average performance of models on evaluation data
# # mode = 'eval'
# # plot_aggregated_performance(model_list, mode, tasks, figurePath)
# # # Plot the average performance of models on training data
# # mode = 'train'
# # plot_aggregated_performance(model_list, mode, tasks, figurePath)



########################################################################################################################
# Functional & Structural Correlation  - Individual networks
########################################################################################################################
def compute_functionalCorrelation(model_dir, figurePath, monthsConsidered, mode):

    analysis = clustering.Analysis(model_dir, mode, monthsConsidered, 'rule')
    correlation = analysis.get_dotProductCorrelation()

    if not os.path.exists(folderPath,'functionalCorrelation_npy'):
        os.makedirs(folderPath,'functionalCorrelation_npy')

    np.save(os.path.join(folderPath,'functionalCorrelation_npy'),correlation)

    # Set up the figure
    fig = plt.figure(figsize=(10, 10))

    # Create the main similarity matrix plot
    matrix_left = 0.1
    matrix_bottom = 0.3
    matrix_width = 0.6
    matrix_height = 0.6

    ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
    im = ax_matrix.imshow(correlation, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1)

    # Add title
    subject = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
    ax_matrix.set_title(f'Correlation Function - {subject}', fontsize=22, pad=20)
    # Add x-axis and y-axis labels
    ax_matrix.set_xlabel('Hidden units', fontsize=16, labelpad=15)
    ax_matrix.set_ylabel('Hidden units', fontsize=16, labelpad=15)

    # Remove x and y ticks
    ax_matrix.set_xticks([])  # Disable x-ticks
    ax_matrix.set_yticks([])  # Disable y-ticks

    # Create the colorbar on the right side, aligned with the matrix
    colorbar_left = matrix_left + matrix_width + 0.02
    colorbar_width = 0.03

    ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Correlation', fontsize=18, labelpad=0)

    # # Set the title above the similarity matrix, centered
    # if mode == 'Training':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
    # elif mode == 'Evaluation':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

    # ax_matrix.set_title(title, fontsize=14, pad=20)
    # Save the figure with a tight bounding box to ensure alignment
    # save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNNmodels\\Visuals\\Similarity\\finalReport',
    #                          model_dir.split("\\")[-1] + '_' + 'Similarity' + '.png')
    # save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\Visuals\\CorrelationFunction\\BarnaModels',
    #                          model_dir.split("\\")[-1] + '_' + 'CorrelationFunction' + '.png')
    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + 'functionalCorrelation' + '.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()

def compute_structuralCorrelation(model_dir, figurePath, monthsConsidered, mode):

    analysis = clustering.Analysis(model_dir, mode, monthsConsidered, 'rule')
    correlationRecurrent = analysis.easy_connectivity_plot_recurrentWeightsOnly(model_dir)
    # correlationExcitatoryGates = analysis.easy_connectivity_plot_excitatoryGatedWeightsOnly(model_dir)
    # correlationInhibitoryGates = analysis.easy_connectivity_plot_inhibitoryGatedWeightsOnly(model_dir)

    if not os.path.exists(folderPath,'structuralCorrelation_npy'):
        os.makedirs(folderPath,'structuralCorrelation_npy')

    np.save(os.path.join(folderPath,'structuralCorrelation_npy'),correlationRecurrent)

    # Set up the figure
    fig = plt.figure(figsize=(10, 10))

    # Create the main similarity matrix plot
    matrix_left = 0.1
    matrix_bottom = 0.3
    matrix_width = 0.6
    matrix_height = 0.6

    ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
    im = ax_matrix.imshow(correlationRecurrent, cmap='coolwarm', interpolation='nearest', vmin=-1, vmax=1) # info: change here

    # Add title
    subject = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
    ax_matrix.set_title(f'correlationRecurrent - {subject}', fontsize=22, pad=20) # info: change here

    # Add x-axis and y-axis labels
    ax_matrix.set_xlabel('Hidden weights', fontsize=16, labelpad=15)
    ax_matrix.set_ylabel('Hidden weights', fontsize=16, labelpad=15)

    # Remove x and y ticks
    ax_matrix.set_xticks([])  # Disable x-ticks
    ax_matrix.set_yticks([])  # Disable y-ticks

    # Create the colorbar on the right side, aligned with the matrix
    colorbar_left = matrix_left + matrix_width + 0.02
    colorbar_width = 0.03

    ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('correlationRecurrent', fontsize=18, labelpad=0) # info: change here

    # # Set the title above the similarity matrix, centered
    # if mode == 'Training':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
    # elif mode == 'Evaluation':
    #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'

    # ax_matrix.set_title(title, fontsize=14, pad=20)
    # Save the figure with a tight bounding box to ensure alignment
    # save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNNmodels\\Visuals\\Similarity\\finalReport',
    #                          model_dir.split("\\")[-1] + '_' + 'Similarity' + '.png')
    # save_path = os.path.join(
    #     'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\Visuals\\CorrelationStructure\\BarnaModels',
    #     model_dir.split("\\")[-1] + '_' + 'CorrelationStructure' + '.png')
    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + 'correlationRecurrent' + '.png'),
                format='png', dpi=300, bbox_inches='tight') # info: change here
    plt.show()



########################################################################################################################
# Execute all functions  - Individual networks
########################################################################################################################
mode = 'Evaluation'

if mode == 'Evaluation':
    months = model_list[0].split('_')[-1].split('-')

monthsConsidered = []
for i in range(int(months[0]), int(months[-1]) + 1):
    monthsConsidered.append(str(i))

for model_dir in model_list:
    # Load hp
    currentHP = Tools.load_hp(model_dir)
    # Assign a color to each task
    _rule_color = {
        'DM': 'green',
        'DM_Anti': 'olive',
        'EF': 'forest green',
        'EF_Anti': 'mustard',
        'RP': 'tan',
        'RP_Anti': 'brown',
        'RP_Ctx1': 'lavender',
        'RP_Ctx2': 'aqua',
        'WM': 'bright purple',
        'WM_Anti': 'green blue',
        'WM_Ctx1': 'blue',
        'WM_Ctx2': 'indigo'
    }
    rule_color = {k: 'xkcd:' + v for k, v in _rule_color.items()}

    # # Plot improvement of performance over iterating evaluation and training steps
    plot_performanceprogress_eval_BeRNN(model_dir, figurePath_performance)
    plot_performanceprogress_train_BeRNN(model_dir, figurePath_performance)
    # # Compute task variance
    # analysis = clustering.Analysis(model_dir, mode, monthsConsidered, 'rule')
    # analysis.plot_variance(model_dir, figurePath_taskVariance, mode) # Normalized Task Variance - Individual networks
    # # Compute functional and structural Correlation through normalized and centered dot product
    compute_functionalCorrelation(model_dir, figurePath_functionalCorrelation, monthsConsidered, mode)
    compute_structuralCorrelation(model_dir, figurePath_structuralCorrelation, monthsConsidered, mode)



########################################################################################################################
# Topological Marker Analysis - RNN
########################################################################################################################
import networkx as nx
import numpy as np
import os
from Network import Model
import tensorflow as tf

folder = '\\barnaModels\\cascade2\\beRNN_05'
folderPath = 'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels' + folder
files = os.listdir(folderPath)
model_list = []
for file in files:
    # if any(include in file for include in ['Model_1_', 'Model_6_']):
    model_list.append(os.path.join(folderPath,file))

def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix, 0)
    return matrix_thresholded

for model_dir in model_list:
    model = Model(model_dir)
    with tf.Session() as sess:
        model.restore()
        # get all connection weights and biases as tensorflow variables
        w_rec = sess.run(model.w_rec)

    # Define a threshold (you can experiment with this value)
    threshold = 0.2  # Example threshold
    w_rec_thresholded = apply_threshold(w_rec, threshold)

    # Function to apply a threshold to the matrix
    G = nx.from_numpy_array(w_rec_thresholded)

    # The degree of a node in a graph is the count of edges connected to that node. For each node, it represents the number of
    # direct connections (or neighbors) it has within the graph.
    degrees = nx.degree(G) # For calculating node degrees. attention: transform into graph then apply stuff
    # Betweenness centrality quantifies the importance of a node based on its position within the shortest paths between other nodes.
    betweenness = nx.betweenness_centrality(G) # For betweenness centrality.
    # Closeness centrality measures the average distance from a node to all other nodes in the network.
    closeness = nx.closeness_centrality(G) # For closeness centrality.
    # average_path_length = nx.average_shortest_path_length(G) # For average path length. attention: Graph is disconnected after thresholding therefore not possible

    # Optionally calculate averages of node-based metrics
    avg_degree = np.mean(list(dict(G.degree()).values()))
    avg_betweenness = np.mean(list(betweenness.values()))
    avg_closeness = np.mean(list(closeness.values()))

    # Graph Density: Measures how many edges exist compared to the maximum possible. This can give you a sense of how interconnected the network is.
    # Max 1; Min 0 Everything or Nothing is connected
    density = nx.density(G)
    # Assortativity: Measures the tendency of nodes to connect to other nodes with similar degrees.
    # A positive value means that high-degree nodes tend to connect to other high-degree nodes. Around 0 no relevant correlation
    assortativity = nx.degree_assortativity_coefficient(G)
    # Transitivity (Global Clustering Coefficient): Measures the likelihood that the adjacent nodes of a node are connected.
    # It’s an indicator of local clustering. 0-1 ; 1 every node that has two neighbours are also connected to each other
    transitivity = nx.transitivity(G)
    # Average Clustering Coefficient: Provides the average of the clustering coefficient (local) for all nodes.
    # It gives you a sense of how well nodes tend to form clusters. Similar to Transitivity
    avg_clustering = nx.average_clustering(G)
    # Largest Connected Component: The size (number of nodes) in the largest connected subgraph of the network.
    # This is particularly important in sparse graphs after thresholding.
    largest_cc = len(max(nx.connected_components(G), key=len))

    metrics = {
        "degrees": degrees,
        "betweenness": betweenness,
        "closeness": closeness,
        "avg_degree": avg_degree,
        "avg_betweenness": avg_betweenness,
        "avg_closeness": avg_closeness,
        "density": density,
        "assortativity": assortativity,
        "transitivity": transitivity,
        "avg_clustering": avg_clustering,
        "largest_cc": largest_cc
    }


    # Define the output directory and file name
    modelName = model_dir.split('\\')[0]
    output_directory = os.path.join(folderPath,'topologicalMarkers_threshold_' + str(threshold))

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the metrics dictionary as a .npy file
    output_file = os.path.join(output_directory, f'topologicalMarkers_threshold_{threshold}_{modelName}.npy')
    np.save(output_file, metrics, allow_pickle=True)

    print(f"Network measures saved to: {output_file}")







# ########################################################################################################################
# # info: Statistical Comparison of topological markers and networks
# ########################################################################################################################
# from scipy.stats import wilcoxon
# stat, p_value = wilcoxon(global_marker_network1, global_marker_network2)
#
# # If distributions of top. Markers are normally distributed:
# from scipy.stats import ttest_ind
# t_stat, p_value = ttest_ind(marker1_participant1_distribution, marker2_participant2_distribution)
#
# # If distribution is not normally distributed or has extreme outliers
# from scipy.stats import mannwhitneyu
# stat, p_value = mannwhitneyu(marker1_participant1_distribution, marker2_participant2_distribution)
#
# # Probably the best choice for two groups of top markers from two groups of network conditions
# from scipy.stats import wilcoxon
# stat, p_value = wilcoxon(marker1_participant1_distribution, marker2_participant2_distribution)
#
# # Potentially tool to multiply the amount of samples trough bootstrapping
# import numpy as np
# def bootstrap(data, n_resamples=1000):
#     resamples = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
#     return resamples
#
# bootstrap_dist1 = bootstrap(path_lengths_network1)
# bootstrap_dist2 = bootstrap(path_lengths_network2)
#
# # info: If all topological markers should be compared to each other
# # Graph Edit Distance: Measures the similarity between two networks based on the number of changes
# # (insertions, deletions, substitutions) required to transform one graph into another.
#
# graph_edit_dist = nx.graph_edit_distance(G1, G2)
#
# eigenvals_G1 = np.linalg.eigvals(nx.laplacian_matrix(G1).A)
# eigenvals_G2 = np.linalg.eigvals(nx.laplacian_matrix(G2).A)
#
# # Percentage change of topological markers (no distribution necessary)
# def percentage_change(marker1, marker2):
#     return ((marker2 - marker1) / marker1) * 100
#
# # Example of comparing density between two networks
# density_model_1 = 0.15
# density_model_2 = 0.25
#
# percent_diff = percentage_change(density_model_1, density_model_2)
# print(f"Percentage Change in Density: {percent_diff:.2f}%")
#
# # Ratio comparison of topological markers (no distribution necessary)
# def ratio_comparison(marker1, marker2):
#     return marker2 / marker1
#
# # Example comparing transitivity between two networks
# transitivity_model_1 = 0.4
# transitivity_model_2 = 0.6
#
# ratio = ratio_comparison(transitivity_model_1, transitivity_model_2)
# print(f"Ratio of Transitivity: {ratio:.2f}")
