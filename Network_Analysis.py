########################################################################################################################
# info: Network Analysis
########################################################################################################################
# Analyze the traine dmodels for their perfromance on training and test data. Investigate the networks for topological
# markers. These markers represent the core and essence of the whole project. It is the objective to find unambiguously
# and reproducable markers for individual models/networks that sould help to distinguish between participants, their traits
# and diseases. Marker are also expected to change over time with evolving traits and diseases, potentially provding a
# dynamical tool to track the course of a disease.
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
import os
# import random
import numpy as np
# import matplotlib
# matplotlib.use('WebAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'wxAgg'
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import tensorflow as tf

from analysis import clustering #, variance
# from NETWORK import Model
import Tools
from Tools import rule_name

########################################################################################################################
# Pre-Allocation of variables and models to be examined
########################################################################################################################
folderPath = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_05_HPT02'
figurePath = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\Visuals\\Performance\\BeRNN_05_HPT02'
files = os.listdir(folderPath)

selected_hp_keys = ['rnn_type', 'activation', 'tau', 'dt', 'sigma_rec', 'sigma_x', 'w_rec_init', 'l1_h', 'l2_h', \
                    'l1_weight', 'l2_weight', 'l2_weight_init', 'learning_rate', 'n_rnn', 'c_mask_responseValue', 'monthsConsidered']  # Replace with the keys you want

model_list = []
for file in files:
    # if any(include in file for include in ['Model_1']):
    model_list.append(os.path.join(folderPath,file))

model_list = ['W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\BeRNN_05_HPT02\\Model_1_BeRNN_05_Month_2-7']

########################################################################################################################
# Performance
########################################################################################################################
# Note to visualization of training and test performance: The test data gives for maxsteps of 1e7 5000 performance data
# points, each representing 800 evaluated trials. The training data gives for maxsteps of 1e7 25000 performance data points,
# each representing 40 trained trials. So I should gather 5 data points of the training data to have the same smoothness
# in the plots, window size = 5
# #######################################################################################################################
def plot_performanceprogress_eval_BeRNN(model_dir, rule_plot=None):
    # Plot Evaluation Progress
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
    trials = log['trials']

    fs = 14  # fontsize
    fig_eval = plt.figure(figsize=(14, 6))
    ax = fig_eval.add_axes([0.1, 0.4, 0.6, 0.5])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    x_plot = np.array(trials) / 1000  # scale the x-axis right
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

    lg = fig_eval.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.15),
                         # co: first value influences horizontal position of legend
                         fontsize=fs, labelspacing=0.3, loc=6, frameon=False)
    plt.setp(lg.get_title(), fontsize=fs)
    # plt.title(model_dir.split("\\")[-1]+'_EVALUATION.png') # info: Add title
    plt.title('_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_EVALUATION', fontsize=16)  # info: Add title
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

    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_EVALUATION.png'), format='png', dpi=300)
    plt.show()

def plot_performanceprogress_train_BeRNN(model_dir, rule_plot=None):
    # Plot Training Progress
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
    trials = log['trials']  # III: There is an entry every 40 trials for each task

    fs = 14  # fontsize
    fig_train = plt.figure(figsize=(14, 6))
    ax = fig_train.add_axes([0.1, 0.4, 0.6, 0.5])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    x_plot = (np.array(trials)) / 1000  # scale the x-axis right
    if rule_plot is None:
        rule_plot = hp['rules']

    # def smoothed(data):
    #     return np.cumsum(data) / np.arange(1, len(data) + 1)

    def smoothed(data, window_size):
        smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
        # Calculate how many points we need to add to match the length of the original data
        padding_length = len(data) - len(smoothed_data)
        if padding_length > 0:
            last_value = smoothed_data[-1]
            smoothed_data = np.concatenate((smoothed_data, [last_value] * padding_length))
        return smoothed_data

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        # line = ax.plot(x_plot, np.log10(log['cost_' + 'WM'][::2]), color=rule_color[rule])
        y_cost = log['cost_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]
        y_perf = log['perf_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]

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

    lg = fig_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.15),
                          # co: first value influences horizontal position of legend
                          fontsize=fs, labelspacing=0.3, loc=6, frameon=False)
    plt.setp(lg.get_title(), fontsize=fs)

    plt.title('_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING', fontsize=16)

    plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_TRAINING.png'), format='png', dpi=300)

    plt.show()


for model_dir in model_list:
    # Load hp
    currentHP = Tools.load_hp(model_dir)

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
    rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}

    # Plot improvement of performance over iterating evaluation steps
    plot_performanceprogress_eval_BeRNN(model_dir)
    # Plot improvement of performance over iterating training steps
    plot_performanceprogress_train_BeRNN(model_dir)

########################################################################################################################
# Clustering
########################################################################################################################
def compute_n_cluster(model_dir, mode, monthsConsidered):
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)
    # try:
    analysis = clustering.Analysis(model_dir, mode, monthsConsidered, 'rule')
    # Plots from instance methods in class
    # analysis.plot_cluster_score(show)
    analysis.plot_variance(model_dir, mode)
    analysis.plot_similarity_matrix(model_dir, mode)
    # analysis.plot_2Dvisualization(model_dir, mode, show)
    # analysis.plot_example_unit(show)
    analysis.plot_connectivity_byclusters(model_dir, mode)

    log['n_cluster'] = analysis.n_cluster
    log['model_dir'] = model_dir
    Tools.save_log(log)
    # except IOError:
    #

for model_dir in model_list:
    months = model_dir.split('_')[-1].split('-')
    monthsConsidered = []
    for i in range(int(months[0]), int(months[1]) + 1):
        monthsConsidered.append(str(i))
    # Load hp
    currentHP = Tools.load_hp(model_dir)

    mode = 'Training'
    compute_n_cluster(model_dir, mode, monthsConsidered)
    mode = 'Evaluation'
    compute_n_cluster(model_dir, mode, monthsConsidered)


