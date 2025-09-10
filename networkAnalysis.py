########################################################################################################################
# head: networkAnalysis ################################################################################################
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import pandas as pd
import glob
import shutil

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from scipy.stats import ttest_ind
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from pathlib import Path

plt.ioff()  # prevents windows to pop up when figs and plots are created

from analysis import clustering  # , variance
import tools
from tools import rule_name


selected_hp_keys = ['participant', 'rnn_type', 'data', 'multiLayer', 'n_rnn_per_layer', 'activations_per_layer',
                    'activation', 'optimizer', 'loss_type', 'batch_size', 'l1_h', 'l2_h', 'l1_weight', 'l2_weight',
                    'learning_rate', 'learning_rate_mode', 'n_rnn', 'sigma_rec', 'sigma_x', 'w_rec_init',
                    'c_mask_responseValue', 'errorBalancingValue', 'p_weight_train',
                    'w_mask_value']  # Replace with the keys you want info: 'data' only exists from 15.01.25 on

rule_color = {
    # **DM tasks (Dark Purple - High Contrast)**
    'DM': '#0d0a29',  # Deep Black-Purple
    'DM_Anti': '#271258',  # Dark Blue-Purple

    # **EF tasks (Purple-Pink Family - High Contrast)**
    'EF': '#491078',  # Muted Indigo
    'EF_Anti': '#671b80',  # Dark Magenta-Purple

    # **RP tasks (Pink/Red Family - High Contrast)**
    'RP': '#862781',  # Rich Magenta
    'RP_Anti': '#a6317d',  # Strong Pink
    'RP_Ctx1': '#c53c74',  # Bright Pinkish-Red
    'RP_Ctx2': '#e34e65',  # Vivid Red

    # **WM tasks (Red-Orange/Yellow Family - High Contrast)**
    'WM': '#f66c5c',  # Warm Reddish-Orange
    'WM_Anti': '#fc9065',  # Vibrant Orange
    'WM_Ctx1': '#feb67c',  # Pastel Orange
    'WM_Ctx2': '#fdda9c'  # Light Yellow
}

comparison = False

def smoothed(data, window_size):
    smoothed_data = np.convolve(data, np.ones(window_size) / window_size, mode='valid')
    # Calculate how many points we need to add to match the length of the original data
    padding_length = len(data) - len(smoothed_data)
    if padding_length > 0:
        last_value = smoothed_data[-1]
        smoothed_data = np.concatenate((smoothed_data, [last_value] * padding_length))
    return smoothed_data

# Create the legend figure
def create_legend_image():
    """ Generate and return a properly formatted legend image. """
    legend_fig = plt.figure(figsize=(2, 0.6))  # Decrease figure size
    legend_ax = legend_fig.add_subplot(111)
    legend_ax.axis("off")

    ordered_rules = list(currentHP['rule_prob_map'].keys())

    # Create handles with appropriate transparency
    rule_handles = [
        plt.Line2D([0], [0], color=rule_color[r], lw=3, alpha=1.0 if currentHP['rule_prob_map'][r] > 1e-10 else 0.5)
        for r in ordered_rules
    ]

    # Labels remain in the same order
    rule_labels = [rule_name[r] for r in ordered_rules]

    # Add legend with optimized spacing
    legend = legend_ax.legend(
        handles=rule_handles, labels=rule_labels, ncol=2, loc="center", fontsize=5,
        frameon=True, edgecolor="black",
        columnspacing=1, handletextpad=0.7
    )
    legend.get_title().set_fontsize(10)  # Slightly larger title

    # Ensure text fits properly
    plt.tight_layout()

    # Convert figure to an image (Ensure full legend is captured)
    canvas = FigureCanvas(legend_fig)
    canvas.draw()

    maskDirectory = os.path.join(os.getcwd(), 'pngs', 'legend.png')
    # Save as buffer to prevent cropping
    legend_fig.savefig(maskDirectory, format='png', dpi=200, bbox_inches='tight', pad_inches=0.05, transparent=True)

    # Load the properly saved image
    legend_img = plt.imread(maskDirectory)

    # Close the figure
    plt.close(legend_fig)

    return legend_img

def fig_to_array(figure):
    """ Convert Matplotlib figure to a NumPy array (RGB image). """
    canvas = FigureCanvas(figure)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    # plt.close(fig)
    return img

def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix,0)
    return matrix_thresholded

def define_data_folder(split_parts):
    # Predefined categories or flags to look for
    possible_keys = [
        'highDim', 'lowDim',
        'correctOnly', 'correctOnly_3stimTC', '3stimTC', 'correctOnly_timeCompressed',
        'lowCognition', 'noCognition', 'timeCompressed'
    ]

    # Filter parts that match any known key
    found_keys = [key for key in possible_keys if key in split_parts]

    # If at least one key is found, join them with underscores
    if found_keys:
        data_folder = 'data_' + '_'.join(found_keys)
    else:
        data_folder = 'data_unknown'

    return data_folder


########################################################################################################################
# Performance - Individual network
########################################################################################################################
# Note to visualization of training and test performance: The test data gives for maxsteps of 1e7 5000 performance data
# points, each representing 800 evaluated trials. The training data gives for maxsteps of 1e7 25000 performance data points,
# each representing 40 trained trials. So I should gather 5 data points of the training data to have the same smoothness
# in the plots, window size = 5
########################################################################################################################
def plot_performanceprogress_test_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Evaluation Progress
    log = tools.load_log(model_dir)
    # log = tools.load_log(currentModelDirectory)
    hp = tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']
    x_plot = np.array(trials) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_eval = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = fig_eval.add_axes([0.12, 0.1, 0.75, 0.65])  # [left, bottom, width, height]
    lines = list()
    labels = list()

    # if rule_plot == None:
    #     # rule_plot = hp['rules']

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        line = ax.plot(x_plot, log['perf_' + rule], color=rule_color[rule], linewidth=3)
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)

    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 80000])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')  # info: Add title

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_test.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_test.png'), format='png', dpi=300)

    # plt.show()
    img_eval = fig_to_array(fig_eval)
    plt.close(fig_eval)

    return img_eval

def plot_performanceprogress_train_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Training Progress
    log = tools.load_log(model_dir)
    hp = tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']  # info: There is an entry every 40 trials for each task
    x_plot = (np.array(trials)) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    fig_train = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = fig_train.add_axes([0.12, 0.1, 0.75, 0.65])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    # if rule_plot is None:
    # rule_plot = hp['rules']
    # rule_plot = ['DM', 'DM_Anti']

    for i, rule in enumerate(rule_plot):
        y_perf = log['perf_train_' + rule][::int((len(log['perf_train_' + rule]) / len(x_plot)))][:len(x_plot)]

        window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation

        # y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        y_perf_smoothed = smoothed(y_perf, window_size=window_size)

        # Ensure the lengths match
        y_perf_smoothed = y_perf_smoothed[:len(x_plot)]

        line = ax.plot(x_plot, y_perf_smoothed, color=rule_color[rule], linewidth=3)

        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    # lg = perf_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.2),
    #                       fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    # plt.setp(lg.get_title(), fontsize=fs)

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_train.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_train.png'), format='png', dpi=300)

    # plt.show()
    img_train = fig_to_array(fig_train)
    plt.close(fig_train)

    return img_train

def plot_cost_test_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Evaluation Progress
    log = tools.load_log(model_dir)
    # log = tools.load_log(currentModelDirectory)
    hp = tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']
    x_plot = np.array(trials) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    cost_eval = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = cost_eval.add_axes([0.12, 0.1, 0.75, 0.65])  # [left, bottom, width, height]
    lines = list()
    labels = list()

    # if rule_plot == None:
    #     # rule_plot = hp['rules']

    # Collect all log-transformed costs for scaling
    all_costs_log = ([np.log10(np.array(log['cost_train_' + rule]) + 1e-8) for rule in rule_plot]
                     + [np.log10(np.array(log['cost_' + rule]) + 1e-8) for rule in rule_plot])

    # Determine global min and max on the log-transformed scale
    cost_log_min = min(np.min(c) for c in all_costs_log)
    cost_log_max = max(np.max(c) for c in all_costs_log)

    for i, rule in enumerate(rule_plot):
        # co: add [::2] if you want to have only every second validation values
        y_cost = log['cost_' + rule]
        # Safe log transform (shift to avoid log(0))
        y_cost_log = np.log10(list(map(lambda x: x + 1e-8, y_cost)))
        # Normalize log costs
        y_cost_log_norm = (y_cost_log - cost_log_min) / (cost_log_max - cost_log_min) # fix: It seems there is some issue here

        line = ax.plot(x_plot, y_cost_log_norm, color=rule_color[rule], linewidth=3)
        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    # ax.set_xlim([0, 80000])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Normalized Log10(Cost)', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')  # info: Add title

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_test.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_test.png'), format='png', dpi=300)

    # plt.show()
    img_eval = fig_to_array(cost_eval)
    plt.close(cost_eval)

    return img_eval

def plot_cost_train_BeRNN(model_dir, figurePath_overview, model, figurePath, rule_plot=None):
    # Plot Training Progress
    log = tools.load_log(model_dir)
    hp = tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    trials = log['trials']  # info: There is an entry every 40 trials for each task
    x_plot = (np.array(trials)) / 1000  # scale the x-axis right

    fs = 18  # fontsize
    cost_train = plt.figure(figsize=(8, 6))
    # ax = fig_eval.add_axes([0.315, 0.1, 0.4, 0.5])  # co: third value influences width of cartoon
    ax = cost_train.add_axes([0.12, 0.1, 0.75, 0.65])  # co: third value influences width of cartoon
    lines = list()
    labels = list()

    # if rule_plot is None:
    # rule_plot = hp['rules']

    # Collect all log-transformed costs for scaling
    all_costs_log = ([np.log10(np.array(log['cost_train_' + rule]) + 1e-8) for rule in rule_plot]
                     + [np.log10(np.array(log['cost_' + rule]) + 1e-8) for rule in rule_plot])

    # Determine global min and max on the log-transformed scale
    cost_log_min = min(np.min(c) for c in all_costs_log)
    cost_log_max = max(np.max(c) for c in all_costs_log)

    for i, rule in enumerate(rule_plot):
        # y_cost = log['cost_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]
        y_cost = log['cost_train_' + rule]
        y_cost_smoothed = smoothed(y_cost, window_size=5)[:len(trials)]

        # window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation
        # y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        # Ensure the lengths match
        # y_cost_smoothed = y_cost_smoothed[:len(x_plot)]
        # Safe log transform (shift to avoid log(0))
        y_cost_log = np.log10(list(map(lambda x: x + 1e-8, y_cost_smoothed)))
        # Normalize log costs
        y_cost_log_norm = (y_cost_log - cost_log_min) / (cost_log_max - cost_log_min)

        # line = ax.plot(x_plot, np.log10(y_cost_smoothed), color=rule_color[rule])
        line = ax.plot(x_plot, y_cost_log_norm, color=rule_color[rule], linewidth=3)

        lines.append(line[0])
        labels.append(rule_name[rule])

    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (*1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Normalized Log10(Cost)', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    ax.grid(True, linestyle='--', alpha=0.6)  # Add grid with dashed lines
    ax.spines["top"].set_linewidth(2)  # Thicker top axis
    ax.spines["right"].set_linewidth(2)  # Thicker right axis
    ax.spines["bottom"].set_linewidth(2)  # Thicker bottom axis
    ax.spines["left"].set_linewidth(2)  # Thicker left axis

    # lg = perf_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.2),
    #                       fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    # plt.setp(lg.get_title(), fontsize=fs)

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_train.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_train.png'), format='png', dpi=300)

    # plt.close()
    img_train = fig_to_array(cost_train)
    plt.close(cost_train)

    return img_train


########################################################################################################################
# Functional & Structural Correlation  - Individual networks
########################################################################################################################
def compute_functionalCorrelation(model_dir, monthsConsidered, mode, figurePath, analysis):
    correlation = analysis.get_dotProductCorrelation()
    # path = os.path.join(figurePath,'functionalCorrelation_npy')

    # if not os.path.exists(path):
    #     os.makedirs(path)

    modelName = model_dir.split('\\')[-1]
    # np.save(os.path.join(figurePath,f'{modelName}_functionalCorrelation'),correlation)

    # Set up the figure
    corr_fig = plt.figure(figsize=(8, 8))

    # Create the main similarity matrix plot
    matrix_left = 0.11
    matrix_bottom = 0.1
    matrix_width = 0.75
    matrix_height = 0.75

    ax_matrix = corr_fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
    im = ax_matrix.imshow(correlation, cmap='magma', interpolation='nearest', vmin=-1, vmax=1)

    # Add title
    model = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
    ax_matrix.set_title(f'{model}', fontsize=26, fontweight='bold', pad=20)
    # Add x-axis and y-axis labels
    ax_matrix.set_xlabel('Hidden units', fontsize=24, labelpad=15)
    ax_matrix.set_ylabel('Hidden units', fontsize=24, labelpad=15)

    # Remove x and y ticks
    ax_matrix.set_xticks([])  # Disable x-ticks
    ax_matrix.set_yticks([])  # Disable y-ticks

    # Create the colorbar on the right side, aligned with the matrix
    colorbar_left = matrix_left + matrix_width + 0.02
    colorbar_width = 0.04

    ax_cb = corr_fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Correlation', fontsize=24, labelpad=0)

    # Compute topological marker
    threshold = 0.5  # info: can be adjusted!
    functionalCorrelation_thresholded = apply_threshold(correlation, threshold)

    # Function to apply a threshold to the matrix
    G = nx.from_numpy_array(functionalCorrelation_thresholded)

    # Modularity: Modularity measures the strength of division of a graph into communities (clusters/modules). A higher modularity means
    # many within-cluster edges and few between-cluster edges.
    communities = greedy_modularity_communities(G)
    mod_value = modularity(G, communities)

    # Betweenness centrality quantifies the importance of a node based on its position within the shortest paths between other nodes.
    betweenness = nx.betweenness_centrality(G)  # For betweenness centrality.
    # Optionally calculate averages of node-based metrics
    avg_betweenness = np.mean(list(betweenness.values()))

    # Assortativity: Measures the tendency of nodes to connect to other nodes with similar degrees.
    # A positive value means that high-degree nodes tend to connect to other high-degree nodes. Around 0 no relevant correlation
    assortativity = nx.degree_assortativity_coefficient(G)

    # Show the top Markers within the func Correlation
    ax_matrix.text(0.5, 0.6, f'Modularity: {mod_value:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.5, f'Betweenness: {avg_betweenness:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.4, f'Assortativity: {assortativity:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    return corr_fig, mod_value, avg_betweenness, assortativity
    # plt.show()
    # plt.close()

if __name__ == "__main__":
    ########################################################################################################################
    # Create Overview and topologcial Marker
    ########################################################################################################################
    def figureSceletton(model_column_widths):
        total_cols = sum(model_column_widths)
        fig = plt.figure(figsize=(3 * total_cols, 16))  # Adjust width proportionally
        gs = fig.add_gridspec(5, total_cols, height_ratios=[1, 1, 1, 1, 0.4])

        fig.text(0.4975, 0.955, 'TRAIN', fontsize=12, fontweight='bold',
                 verticalalignment='center', horizontalalignment='center', color='black')
        fig.text(0.4975, 0.52, 'TEST', fontsize=12, fontweight='bold',
                 verticalalignment='center', horizontalalignment='center', color='black')

        axs = [[] for _ in range(5)]  # rows 0 to 4
        current_col = 0

        for model_width in model_column_widths:
            # Row 0: train cost/perf (2 cols)
            axs[0].append(fig.add_subplot(gs[0, current_col:current_col + model_width]))
            # Row 1: train functional correlation (n layers)
            axs[1].extend([fig.add_subplot(gs[1, current_col + i]) for i in range(model_width)])
            # Row 2: test cost/perf (2 cols)
            axs[2].append(fig.add_subplot(gs[2, current_col:current_col + model_width]))
            # Row 3: test functional correlation (n layers)
            axs[3].extend([fig.add_subplot(gs[3, current_col + i]) for i in range(model_width)])

            current_col += model_width

        # Row 4: legend and HP (2 subplots only)
        axs[4] = [
            fig.add_subplot(gs[4, 0:total_cols // 2]),
            fig.add_subplot(gs[4, total_cols // 2:total_cols])
        ]

        return fig, axs

    folder, dataType, participant, batch, months = 'autoCorrelationTest_04', 'highDim_correctOnly_3stimTC', 'beRNN_04', '01', ['month_4', 'month_5', 'month_6']
    _finalPath = Path('C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels', f'{folder}/{dataType}/{participant}/{batch}')
    _data_dir = Path('C:/Users/oliver.frank/Desktop/PyProjects/Data')

    # Define paths
    participant_id = participant  # Change this for each participant
    topMarker_path = os.path.join(_finalPath, 'overviews', 'all_topologicalMarker_files')
    os.makedirs(topMarker_path, exist_ok=True)
    # Create overview folder
    overviewFolder = Path(_finalPath, 'overviews')
    os.makedirs(overviewFolder, exist_ok=True)
    # Create model list for every iteration
    _model_list = os.listdir(_finalPath)
    _model_list = [i for i in _model_list if 'beRNN' in i]

    for _model in _model_list:
        try:
            # Re-Initialize for each model
            model_column_widths = []
            _model_paths = []

            model_dir = os.path.join(_finalPath, _model)
            model_list = sorted([i for i in os.listdir(model_dir) if 'model' in i])

            # Loop through all models per months for each model
            for model in model_list:
                currentModelDirectory = os.path.join(model_dir, model)
                currentHP = tools.load_hp(currentModelDirectory)

                if currentHP.get('multiLayer'):
                    n_layers = len(currentHP['n_rnn_per_layer'])
                else:
                    n_layers = 1

                block_width = max(n_layers, 3)
                model_column_widths.append(block_width)
                _model_paths.append((_model, model))

            # Create dynamic figure
            fig, axs = figureSceletton(model_column_widths)

            # Now process each model and plot
            col_start = 0
            for model_idx, (_model, model) in enumerate(_model_paths):
                model_dir = os.path.join(_finalPath, _model)
                currentModelDirectory = os.path.join(model_dir, model)
                currentHP = tools.load_hp(currentModelDirectory)

                dataFolder = define_data_folder(_model.split('_'))

                data_dir = os.path.join(_data_dir, participant, dataFolder)
                rule_plot = [i for i in currentHP['rule_prob_map'] if currentHP['rule_prob_map'][i] > 0]

                # Plot performance and cost
                img_perf_test = plot_performanceprogress_test_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                img_perf_train = plot_performanceprogress_train_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                img_cost_test = plot_cost_test_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)
                img_cost_train = plot_cost_train_BeRNN(currentModelDirectory, "", model, None, rule_plot=rule_plot)

                # Plot correlations
                n_layers = len(currentHP['n_rnn_per_layer']) if currentHP.get('multiLayer') else 1
                funcTrainList, funcTestList = [], []
                for layer in range(n_layers):
                    try:
                        analysis_train = clustering.Analysis(data_dir, currentModelDirectory, layer,  'cosine', 'train', currentHP['monthsConsidered'], 'rule', True)
                        analysis_test = clustering.Analysis(data_dir, currentModelDirectory, layer, 'cosine','test', currentHP['monthsConsidered'], 'rule', True)

                        func_train, *_ = compute_functionalCorrelation(currentModelDirectory, currentHP['monthsConsidered'], 'train', None, analysis_train)
                        func_test, avg_mod_test, avg_betweenness_test, assortativity_test = compute_functionalCorrelation(currentModelDirectory, currentHP['monthsConsidered'], 'test', None, analysis_test)

                        # info: Append test top. Markers into a list and save them in folder #######################################
                        modList = []
                        betweennessList = []
                        assortativityList = []
                        modList.append(avg_mod_test)  # fix: exchange with modularity
                        betweennessList.append(avg_betweenness_test)
                        assortativityList.append(assortativity_test)

                        topMarkerList = [modList, betweennessList, assortativityList]
                        topMarkerNamesList = ['modList', 'betweennessList', 'assortativityList']
                        for i in range(0, len(topMarkerNamesList)):
                            mean_value = np.mean(topMarkerList[i])
                            variance_value = np.var(topMarkerList[i])
                            # mean_variance = np.array([mean_value, variance_value])
                            np.save(os.path.join(topMarker_path, f'{topMarkerNamesList[i]}_{_model}_{model}_layer{layer}.npy'), topMarkerList[i])
                        # info: ####################################################################################################

                        funcTrainList.append(fig_to_array(func_train))
                        funcTestList.append(fig_to_array(func_test))

                        plt.close(func_train)
                        plt.close(func_test)

                    except Exception as e:
                        print(f"Skipping layer {layer} for model {currentModelDirectory} due to error: {e}")
                        # You can optionally append blank images or zero arrays to keep the grid layout consistent
                        placeholder = np.zeros((currentHP['n_rnn_per_layer'][layer], currentHP['n_rnn_per_layer'][layer], 3), dtype=np.uint8)  # black placeholder RGB image
                        funcTrainList.append(placeholder)
                        funcTestList.append(placeholder)

                # Place plots into skeleton
                combined_train = np.concatenate((img_cost_train, img_perf_train), axis=1)
                axs[0][model_idx].imshow(combined_train)
                combined_test = np.concatenate((img_cost_test, img_perf_test), axis=1)
                axs[2][model_idx].imshow(combined_test)

                for i in range(n_layers):
                    axs[1][col_start + i].imshow(funcTrainList[i])
                    axs[3][col_start + i].imshow(funcTestList[i])

                # Clean up axes
                for row in range(4):
                    for i in range(model_column_widths[model_idx]):
                        if row == 0 or row == 2:
                            axs[row][model_idx].axis('off')
                        else:
                            axs[row][col_start + i].axis('off')

                col_start += model_column_widths[model_idx]

                # Add legend
                legend_img = create_legend_image()
                axs[4][0].imshow(legend_img)
                axs[4][0].axis("off")

                # Add hyperparameter box
                hp_lines = [f"{key}: {currentHP[key]}" for key in selected_hp_keys if key in currentHP]
                num_columns = 3
                hp_text_formatted = "\n".join(["   ".join(hp_lines[i:i + num_columns]) for i in range(0, len(hp_lines), num_columns)])
                # fix: fontsize decides size of box and should be individual for number of layers - 3: 9
                axs[4][1].text(0.5, 0.5, hp_text_formatted, fontsize=9, verticalalignment='center',
                               horizontalalignment='center', color='black',
                               bbox=dict(facecolor='white', alpha=1, edgecolor='black'))
                axs[4][1].axis("off")

            plt.tight_layout()
            overview_path = os.path.join(overviewFolder, f'{_model}_OVERVIEW.png')
            plt.savefig(overview_path, dpi=150, bbox_inches='tight')
            # plt.close(fig)

            print(f"Overview saved to: {overview_path}")

        except Exception as e:
            print("An exception occurred with model number: ", model_dir)
            print("Error: ", e)



    # info: ################################################################################################################
    # info: Distribution plot ##############################################################################################
    # info: ################################################################################################################

    # Collect all top Marker files into one destination
    _iterationList = os.listdir(_finalPath) # info: Folder of several iterations for one training batch of one participant
    iterationList = [os.path.join(_finalPath, iteration, 'topologicalMarker')
                     for iteration in _iterationList if 'beRNN' in iteration]

    npy_files = glob.glob(os.path.join(topMarker_path, "*.npy"))
    # Extract markers and months from filenames
    topMarkers = ['assortativityList', 'betweennessList', 'modList']

    num_rows = len(topMarkers)
    num_columns = len(months)

    # Group files based on (marker, month)
    groups = {marker: {month: [] for month in months} for marker in topMarkers}
    for file in npy_files:
        for marker in topMarkers:
            for month in months:
                if marker in file and month in file:
                    groups[marker][month].append(os.path.join(topMarker_path, file))

    # Create figure and axes
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(6, 1.5 * num_rows), sharex=False, sharey=False)

    # Ensure axes is always a 2D array
    if num_rows == 1:
        axes = np.array([axes])
    if num_columns == 1:
        axes = np.expand_dims(axes, axis=1)

    # Directory to save distributions
    distribution_dir = os.path.join(_finalPath, 'overviews', "topologicalMarker_lists")
    os.makedirs(distribution_dir, exist_ok=True)

    # Dictionary to store t-test results
    t_test_results = {}

    # Process and plot distributions
    for row, marker in enumerate(topMarkers):
        t_test_results[marker] = {}

        for col, month in enumerate(months):
            files = groups[marker][month]
            ax = axes[row, col]

            if files:
                all_data = []
                for file in files:
                    try:
                        data = np.load(file)
                        all_data.append(data)
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

                if all_data:
                    # Convert to NumPy arrays and filter out empty data
                    all_data = [np.asarray(arr).flatten() for arr in all_data if arr.size > 0]

                    if all_data:
                        if len(all_data) == 1:
                            valid_data = all_data[0]
                        else:
                            valid_data = np.concatenate(all_data)

                        # Save distribution for later comparisons
                        np.save(os.path.join(distribution_dir, f"{marker}_{month}.npy"), valid_data)

                        mean, variance = np.mean(valid_data), np.var(valid_data)

                        # Plot histogram
                        ax.hist(valid_data, bins=20, density=False, alpha=0.7, color='skyblue', edgecolor='black')
                        ax.axvline(mean, color='red', linestyle='dashed', linewidth=1.5,
                                   label=f'Mean: {mean:.2f}\nVar: {variance:.2f}')
                        ax.set_title(f"{marker} - {month}", fontsize=12)
                        ax.legend(fontsize=6)

                        ax.tick_params(axis='both', which='major', labelsize=6)  # Adjust size as needed
                        ax.tick_params(axis='both', which='minor', labelsize=6)  # Even smaller for minor ticks

                        # Store t-test results
                        if col > 0:
                            prev_files = groups[marker][months[col - 1]]

                            prev_data_list = []
                            for prev_file in prev_files:
                                try:
                                    prev_data_list.append(np.load(prev_file))
                                except Exception as e:
                                    print(f"Error loading {prev_file}: {e}")

                            # Ensure previous data is valid
                            if prev_data_list:
                                prev_data_list = [np.asarray(arr).flatten() for arr in prev_data_list if arr.size > 0]
                                if prev_data_list:
                                    prev_data = np.concatenate(prev_data_list) if len(prev_data_list) > 1 else prev_data_list[0]

                                    if len(prev_data) > 1 and len(valid_data) > 1:
                                        t_stat, p_value = ttest_ind(prev_data, valid_data, equal_var=False)
                                        t_test_results[marker][(months[col - 1], month)] = (t_stat, p_value)

                                        if p_value < 0.05:
                                            ax.annotate(f'* p={p_value:.2e}', xy=(0.5, 0.55), xycoords='axes fraction',
                                                        fontsize=6, fontweight='bold', ha='center', color='red')
                    else:
                        ax.set_title(f"{marker} - {month} (No Valid Data)", fontsize=10)
                else:
                    ax.set_title(f"{marker} - {month} (No Data)", fontsize=10)
            else:
                ax.set_title(f"{marker} - {month} (No Data)", fontsize=10)

    # Adjust layout and save plot
    plt.tight_layout()
    plt.suptitle(f"Distributions for {participant_id}", fontsize=12, fontweight='bold', y=1.02)

    # Save the figure
    plot_path = os.path.join(_finalPath, 'overviews', f"topologicalMarkers_distribution_{participant_id}_{dataType}_{batch}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    # plt.show()

    print(f"Plot saved at: {plot_path}")
    print(f"Distributions saved for participant {participant_id} in {distribution_dir}")


if comparison == True:
    # # info: ################################################################################################################
    # # info: Comparison - Only apply after previous analysis ################################################################
    # # info: ################################################################################################################
    from scipy.stats import ttest_ind, ks_2samp
    import seaborn as sns

    # Define variables for topological marker distribution comparison
    participant1, batch1 = 'beRNN_01', '32'
    participant2, batch2 = 'beRNN_03', '32'

    def load_distributions(distribution_dir,topMarkers,months):
        """
        Load saved distributions for a given participant.
        """
        distributions = {}

        if not os.path.exists(distribution_dir):
            print(f"No distributions found for {distribution_dir}")
            return None

        for file in os.listdir(distribution_dir):
            if file.endswith(".npy"):
                for marker in topMarkers:
                    for month in months:
                        # Safely initialize nested dict
                        if marker not in distributions:
                            distributions[marker] = {}
                        distributions[marker][month] = np.load(os.path.join(distribution_dir, file))

        return distributions

    def compare_participants(dist_1, dist_2, participant_1, participant_2, destination_dir):
        """
        Compare the distributions of two participants and display significance.
        """
        p_values = {}  # Store p-values for visualization

        for marker in dist_1.keys():
            p_values[marker] = {}

            for month in dist_1[marker].keys():
                if marker in dist_2 and month in dist_2[marker]:  # Ensure both have data
                    data_1 = dist_1[marker][month]
                    data_2 = dist_2[marker][month]

                    if len(data_1) > 1 and len(data_2) > 1:
                        # Perform statistical tests
                        t_stat, p_ttest = ttest_ind(data_1, data_2, equal_var=False)
                        ks_stat, p_ks = ks_2samp(data_1, data_2)

                        p_values[marker][month] = min(p_ttest, p_ks)  # Store min p-value
                    else:
                        p_values[marker][month] = 1.0  # No valid comparison

        # Convert to DataFrame for visualization
        p_df = pd.DataFrame(p_values).T  # Transpose so markers are rows, months are columns

        # Prepare text annotations with significance levels
        def format_p_value(p):
            if p < 0.001:
                return f"$\\bf{{{p:.3f}}}$***"  # Bold + ***
            elif p < 0.01:
                return f"$\\bf{{{p:.3f}}}$**"  # Bold + **
            elif p < 0.05:
                return f"$\\bf{{{p:.3f}}}$*"  # Bold + *
            else:
                return f"{p:.3f}"  # No bold

        annotations = p_df.applymap(format_p_value)

        # Plot heatmap of p-values
        plt.figure(figsize=(10, 6))
        ax = sns.heatmap(
            p_df.astype(float),
            annot=annotations,
            fmt="",
            cmap="magma",  # Reverse "magma" so low p-values are lighter
            vmin=0.001,
            vmax=1.0,
            center=0.05,
            cbar_kws={"shrink": 1.0},  # Fix legend error
            annot_kws={"fontsize": 10, "color": "white"},  # Ensure all text is white
        )

        plt.title(f"Statistical Comparison: {participant_1} vs {participant_2}")
        plt.xlabel("Months")
        plt.ylabel("Topological Markers")

        # Save and show the plot
        plot_path = os.path.join(destination_dir, '_topologicalMarkerComparison')
        os.makedirs(plot_path, exist_ok=True)
        plt.savefig(os.path.join(plot_path, f"topMarkerComparison_{participant_1}_{participant_2}.png"), dpi=300, bbox_inches='tight')
        plt.show()


    destination_dir = Path('C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels', f'{folder}/{dataType}')
    os.makedirs(destination_dir, exist_ok=True)

    distributions_dir_participant_01 = f"{destination_dir}\\{participant1}\\{batch1}\\overviews\\topologicalMarker_lists"
    distributions_dir_participant_02 = f"{destination_dir}\\{participant2}\\{batch2}\\overviews\\topologicalMarker_lists"

    dist_1 = load_distributions(distributions_dir_participant_01,topMarkers,months)
    dist_2 = load_distributions(distributions_dir_participant_02,topMarkers,months)

    compare_participants(dist_1, dist_2, participant1, participant2, destination_dir)



# # info: ################################################################################################################
# # info: Structural correlation for single network ######################################################################
# # info: ################################################################################################################
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
# from analysis import clustering
#
# model_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\robustnessTest\highDim_correctOnly\beRNN_01\0\beRNN_01_AllTask_4-6_data_highDim_correctOnly_iteration1_LeakyRNN_diag_256_softplus'
# data_dir = os.path.join(Path('C:/Users/oliver.frank/Desktop/PyProjects/Data'), 'beRNN_01', 'data_highDim_correctOnly')
# mode = 'test' # 'train' 'test'
#
# figurePath = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\robustnessTest\highDim_correctOnly\beRNN_01\visuals\performance_test\structuralCorrelation'
# monthsConsidered = ['month_4', 'month_5', 'month_6']
# layer = 1
#
# def compute_structuralCorrelation(model_dir, figurePath, monthsConsidered, mode):
#     for month in monthsConsidered:
#         analysis = clustering.Analysis(data_dir, os.path.join(model_dir, f'model_{month}'), layer, 'cosine', mode, monthsConsidered, 'rule', True)
#         correlationRecurrent = analysis.easy_connectivity_plot_recurrentWeightsOnly(os.path.join(model_dir, f'model_{month}'))
#         # correlationExcitatoryGates = analysis.easy_connectivity_plot_excitatoryGatedWeightsOnly(model_dir)
#         # correlationInhibitoryGates = analysis.easy_connectivity_plot_inhibitoryGatedWeightsOnly(model_dir)
#
#         path = os.path.join(figurePath, 'structuralCorrelation_npy')
#
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         correlationNames = ['CorrelationRecurrent']  # , 'CorrelationInhibitoryGates', 'CorrelationExcitatoryGates']
#
#         correlationDict = {'CorrelationRecurrent': correlationRecurrent}
#         # 'CorrelationInhibitoryGates': correlationInhibitoryGates,
#         # 'CorrelationExcitatoryGates': correlationExcitatoryGates}
#
#         for correlationName in correlationNames:
#             modelName = model_dir.split('beRNN_')[-1]
#             np.save(os.path.join(path, f'structural{correlationName}_{modelName}_{mode}.npy'),
#                     correlationDict[correlationName])
#
#             # Set up the figure
#             fig = plt.figure(figsize=(10, 10))
#
#             # Create the main similarity matrix plot
#             matrix_left = 0.1
#             matrix_bottom = 0.3
#             matrix_width = 0.6
#             matrix_height = 0.6
#
#             ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
#             im = ax_matrix.imshow(correlationDict[correlationName], cmap='coolwarm', interpolation='nearest', vmin=-1,
#                                   vmax=1)  # info: change here
#
#             # Add title
#             # subject = '_'.join(model_dir.split("\\")[-1].split('_')[0:4])
#             ax_matrix.set_title(f'Structural Correlation - {month}', fontsize=22, pad=20)  # info: change here
#
#             # Add x-axis and y-axis labels
#             ax_matrix.set_xlabel('Hidden weights', fontsize=16, labelpad=15)
#             ax_matrix.set_ylabel('Hidden weights', fontsize=16, labelpad=15)
#
#             # Remove x and y ticks
#             ax_matrix.set_xticks([])  # Disable x-ticks
#             ax_matrix.set_yticks([])  # Disable y-ticks
#
#             # Create the colorbar on the right side, aligned with the matrix
#             colorbar_left = matrix_left + matrix_width + 0.02
#             colorbar_width = 0.03
#
#             ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
#             cb = plt.colorbar(im, cax=ax_cb)
#             cb.set_ticks([-1, 1])
#             cb.outline.set_linewidth(0.5)
#             cb.set_label('Correlation', fontsize=18, labelpad=0)  # info: change here
#
#             # # Set the title above the similarity matrix, centered
#             # if mode == 'Training':
#             #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TRAINING'
#             # elif mode == 'Evaluation':
#             #     title = '_'.join(model_dir.split("\\")[-1].split('_')[0:4]) + '_TEST'
#
#             # ax_matrix.set_title(title, fontsize=14, pad=20)
#             # Save the figure with a tight bounding box to ensure alignment
#             # save_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNNmodels\\Visuals\\Similarity\\finalReport',
#             #                          model_dir.split("\\")[-1] + '_' + 'Similarity' + '.png')
#             # save_path = os.path.join(
#             #     'W:\\group_csp\\analyses\\oliver.frank\\beRNNmodels\\Visuals\\CorrelationStructure\\BarnaModels',
#             #     model_dir.split("\\")[-1] + '_' + 'CorrelationStructure' + '.png')
#             plt.savefig(os.path.join(figurePath, f'{correlationName}_{modelName}_{month}_{mode}.png'), format='png', dpi=300, bbox_inches='tight')  # info: change here
#
#             plt.show()
#             # plt.close()
#
# compute_structuralCorrelation(model_dir, figurePath, monthsConsidered, mode)


