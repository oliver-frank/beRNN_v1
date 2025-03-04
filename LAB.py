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
plt.ioff() # prevents windows to pop up when figs and plots are created
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.patches import Rectangle, Polygon
from matplotlib.gridspec import GridSpec
import networkx as nx
import gc
# matplotlib.use('WebAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'wxAgg'
# from scipy.stats import ttest_ind
import shutil

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from analysis import clustering #, variance
import Tools
from Tools import rule_name
# from Network import Model
# import tensorflow as tf


selected_hp_keys = ['participant', 'rnn_type', 'data', 'activation', 'optimizer', 'loss_type', 'batch_size', 'l1_h', 'l2_h', 'l1_weight', 'l2_weight',
                    'learning_rate', 'learning_rate_mode', 'n_rnn', 'tau', 'sigma_rec', 'sigma_x', 'w_rec_init', 'c_mask_responseValue', 'p_weight_train', 'w_mask_value'] # Replace with the keys you want info: 'data' only exists from 15.01.25 on

rule_color = {
    # **DM tasks (Dark Purple - High Contrast)**
    'DM':       '#0d0a29',  # Deep Black-Purple
    'DM_Anti':  '#271258',  # Dark Blue-Purple

    # **EF tasks (Purple-Pink Family - High Contrast)**
    'EF':       '#491078',  # Muted Indigo
    'EF_Anti':  '#671b80',  # Dark Magenta-Purple

    # **RP tasks (Pink/Red Family - High Contrast)**
    'RP':       '#862781',  # Rich Magenta
    'RP_Anti':  '#a6317d',  # Strong Pink
    'RP_Ctx1':  '#c53c74',  # Bright Pinkish-Red
    'RP_Ctx2':  '#e34e65',  # Vivid Red

    # **WM tasks (Red-Orange/Yellow Family - High Contrast)**
    'WM':       '#f66c5c',  # Warm Reddish-Orange
    'WM_Anti':  '#fc9065',  # Vibrant Orange
    'WM_Ctx1':  '#feb67c',  # Pastel Orange
    'WM_Ctx2':  '#fdda9c'   # Light Yellow
}

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

def fig_to_array(fig):
    """ Convert Matplotlib figure to a NumPy array (RGB image). """
    canvas = FigureCanvas(fig)
    canvas.draw()
    img = np.array(canvas.renderer.buffer_rgba())
    plt.close(fig)
    return img

def apply_threshold(matrix, threshold):
    # Set all values below the threshold to zero
    matrix_thresholded = np.where(np.abs(matrix) > threshold, matrix,0)  # fix: Can you appply a 20 to 40 % of the strongest connection filter for positive and negative correlations
    return matrix_thresholded


########################################################################################################################
# Performance - Individual network
########################################################################################################################
# Note to visualization of training and test performance: The test data gives for maxsteps of 1e7 5000 performance data
# points, each representing 800 evaluated trials. The training data gives for maxsteps of 1e7 25000 performance data points,
# each representing 40 trained trials. So I should gather 5 data points of the training data to have the same smoothness
# in the plots, window size = 5
########################################################################################################################
def plot_performanceprogress_test_BeRNN(model_dir, figurePath, figurePath_overview, model, rule_plot=None):
    # Plot Evaluation Progress
    log = Tools.load_log(model_dir)
    # log = Tools.load_log(currentModelDirectory)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
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
        # line = ax.plot(x_plot, np.log10(log['cost_' + rule]), color=rule_color[rule])
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

    return fig_eval
    # plt.show()
    # plt.close()

def plot_performanceprogress_train_BeRNN(model_dir, figurePath, figurePath_overview, model, rule_plot=None):
    # Plot Training Progress
    log = Tools.load_log(model_dir)
    hp = Tools.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
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
        # y_cost = log['cost_train_' + rule][::int((len(log['cost_train_' + rule]) / len(x_plot)))][:len(x_plot)]

        y_perf = log['perf_train_' + rule][::int((len(log['perf_train_' + rule]) / len(x_plot)))][:len(x_plot)]

        window_size = 5  # Adjust window_size to smooth less or more, should actually be 20 so that it concolves the same amount of data (800 trials) for one one measure as in evaluation

        # y_cost_smoothed = smoothed(y_cost, window_size=window_size)
        y_perf_smoothed = smoothed(y_perf, window_size=window_size)

        # Ensure the lengths match
        # y_cost_smoothed = y_cost_smoothed[:len(x_plot)]
        y_perf_smoothed = y_perf_smoothed[:len(x_plot)]

        # line = ax.plot(x_plot, np.log10(y_cost_smoothed), color=rule_color[rule])
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

    # lg = fig_train.legend(lines, labels, title='Task', ncol=2, bbox_to_anchor=(0.1, 0.2),
    #                       fontsize=fs, labelspacing=0.3, loc=6, frameon=False) # info: first value influences horizontal position of legend
    # plt.setp(lg.get_title(), fontsize=fs)

    plt.title(model_dir.split("\\")[-1], fontsize=20, fontweight='bold')

    # plt.savefig(os.path.join(figurePath, model_dir.split("\\")[-1] + '_' + model + '_performance_train.png'), format='png', dpi=300)
    # plt.savefig(os.path.join(figurePath_overview, model_dir.split("\\")[-2] + '_' + model + '_train.png'), format='png', dpi=300)

    return fig_train
    # plt.show()
    # plt.close()


########################################################################################################################
# Functional & Structural Correlation  - Individual networks
########################################################################################################################
def compute_functionalCorrelation(model_dir, figurePath, monthsConsidered, mode, analysis):

    correlation = analysis.get_dotProductCorrelation()
    # path = os.path.join(figurePath,'functionalCorrelation_npy')

    # if not os.path.exists(path):
    #     os.makedirs(path)

    modelName = model_dir.split('\\')[-1]
    np.save(os.path.join(figurePath,f'{modelName}_functionalCorrelation'),correlation)

    # Set up the figure
    fig = plt.figure(figsize=(8, 8))

    # Create the main similarity matrix plot
    matrix_left = 0.11
    matrix_bottom = 0.1
    matrix_width = 0.75
    matrix_height = 0.75

    ax_matrix = fig.add_axes([matrix_left, matrix_bottom, matrix_width, matrix_height])
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

    ax_cb = fig.add_axes([colorbar_left, matrix_bottom, colorbar_width, matrix_height])
    cb = plt.colorbar(im, cax=ax_cb)
    cb.set_ticks([-1, 1])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Correlation', fontsize=24, labelpad=0)

    # Compute topological marker
    threshold = 0.5  # can be adjusted
    functionalCorrelation_thresholded = apply_threshold(correlation, threshold)

    # Function to apply a threshold to the matrix
    G = nx.from_numpy_array(functionalCorrelation_thresholded)

    # The degree of a node in a graph is the count of edges connected to that node. For each node, it represents the number of
    # direct connections (or neighbors) it has within the graph.
    degrees = nx.degree(G)  # For calculating node degrees
    # Betweenness centrality quantifies the importance of a node based on its position within the shortest paths between other nodes.
    betweenness = nx.betweenness_centrality(G)  # For betweenness centrality.
    # Optionally calculate averages of node-based metrics
    avg_degree = np.mean(list(dict(G.degree()).values()))
    avg_betweenness = np.mean(list(betweenness.values()))

    # Assortativity: Measures the tendency of nodes to connect to other nodes with similar degrees.
    # A positive value means that high-degree nodes tend to connect to other high-degree nodes. Around 0 no relevant correlation
    assortativity = nx.degree_assortativity_coefficient(G)

    # Show the top Markers within the func Correlation
    ax_matrix.text(0.5, 0.6, f'Degrees: {avg_degree:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.5, f'Betweenness: {avg_betweenness:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    ax_matrix.text(0.5, 0.4, f'Assortativity: {assortativity:.3f}', fontsize=22, color='white', fontweight='bold',
                   ha='center', va='center', transform=ax_matrix.transAxes)
    return fig
    # plt.show()
    # plt.close()


########################################################################################################################
# Create Overview
########################################################################################################################
# Create basic sceleton for all the plots created for each individual model, respectively
def figureSceletton():
    # Create figure with gridspec
    fig = plt.figure(figsize=(19.2, 10.8))
    gs = GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 1])  # Last row is shorter

    fig.text(0.4975, 0.955, 'TRAIN', fontsize=12, fontweight='bold',
             verticalalignment='center', horizontalalignment='center', color='black')
    fig.text(0.4975, 0.57, 'TEST', fontsize=12, fontweight='bold',
             verticalalignment='center', horizontalalignment='center', color='black')

    fig.text(0.375, 0.16, 'Task', fontsize=8, fontweight='bold',
             verticalalignment='center', horizontalalignment='center', color='black')
    fig.text(0.62, 0.16, 'Hyperparameter', fontsize=8, fontweight='bold',
             verticalalignment='center', horizontalalignment='center', color='black')

    # # Define a small triangle (Right-pointing arrow)
    # triangle_size = 0.01  # Adjust size of the triangle
    # triangle_x = [0.395, 0.58, 0.395, 0.58]  # X position (closer to right side)
    # triangle_y = [0.393, 0.393, 0.785, 0.785]  # Y position (adjust based on where you want it)
    #
    # for i in range(0, len(triangle_y)):
    #     triangle = Polygon(
    #         [[triangle_x[i] + triangle_size, triangle_y[i]],  # Triangle tip (rightmost point)
    #          [triangle_x[i] + triangle_size / 2, triangle_y[i] + triangle_size / 2],  # Top-left corner
    #          [triangle_x[i] + triangle_size / 2, triangle_y[i] - triangle_size / 2]],  # Bottom-left corner
    #         closed=True, facecolor='black', edgecolor='black', transform=fig.transFigure, clip_on=False
    #     )
    #
    #     fig.add_artist(triangle)

    # Create subplots for first 4 rows
    axs = [[fig.add_subplot(gs[row, col]) for col in range(3)] for row in range(4)]
    # Create a single merged subplot for the last row
    axs_legend = fig.add_subplot(gs[4, :])

    # Remove all extra spacing for maximum image fill
    plt.subplots_adjust(left=0.7, right=0.92, top=0.98, bottom=0.05, wspace=0, hspace=0.05)

    return fig, axs, axs_legend

# Paths and settings
participant = 'beRNN_03'
trainingNumber = '\\08'
folder = '\\beRNNmodels\\2025_03'
folderPath = 'W:\\group_csp\\analyses\\oliver.frank'
# folderPath = 'C:\\Users\\oliver.frank\\Desktop\\BackUp'
_finalPath = folderPath + folder + trainingNumber

# Create overview folder
overviewFolder = _finalPath + '\\overviews'
os.makedirs(overviewFolder, exist_ok=True)

_model_list = os.listdir(_finalPath)
_model_list = [i for i in _model_list if 'beRNN' in i]

for _model in _model_list:
    try:

        if len(_model.split('_')) == 10:
            dataFolder = '_'.join(_model.split('_')[4:6])
        elif len(_model.split('_')) == 11:
            dataFolder = '_'.join(_model.split('_')[4:7])

        model_dir = os.path.join(_finalPath, _model)
        model_list = os.listdir(model_dir)
        model_list = [i for i in model_list if 'model' in i]

        # Define right model
        data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data', participant, dataFolder)

        figs_to_close = []
        fig, axs, axs_legend = figureSceletton()

        for col, model in enumerate(model_list):
            currentModelDirectory = os.path.join(_finalPath, _model, model)
            currentHP = Tools.load_hp(currentModelDirectory)

            rule_plot = [i for i in currentHP['rule_prob_map'] if currentHP['rule_prob_map'][i] > 0]
            visualsDirectory = os.path.join(currentModelDirectory, 'visuals')
            os.makedirs(visualsDirectory, exist_ok=True)

            # Generate plots
            fig_test = plot_performanceprogress_test_BeRNN(currentModelDirectory, visualsDirectory, "", model, rule_plot=rule_plot)
            fig_train = plot_performanceprogress_train_BeRNN(currentModelDirectory, visualsDirectory, "", model, rule_plot=rule_plot)

            analysis_test = clustering.Analysis(data_dir, currentModelDirectory, 'test', currentHP['monthsConsidered'], 'rule')
            analysis_train = clustering.Analysis(data_dir, currentModelDirectory, 'train', currentHP['monthsConsidered'], 'rule')

            fig_func_test = compute_functionalCorrelation(currentModelDirectory, visualsDirectory, currentHP['monthsConsidered'], 'test', analysis_test)
            fig_func_train = compute_functionalCorrelation(currentModelDirectory, visualsDirectory, currentHP['monthsConsidered'], 'train', analysis_train)

            # Convert to image arrays
            img_test = fig_to_array(fig_test)
            img_train = fig_to_array(fig_train)
            img_func_test = fig_to_array(fig_func_test)
            img_func_train = fig_to_array(fig_func_train)

            # Set subplot sizes to match image aspect ratios without stretching
            axs[0][col].imshow(img_train)
            axs[1][col].imshow(img_func_train)
            axs[2][col].imshow(img_test)
            axs[3][col].imshow(img_func_test)

            # Remove axes, ticks, and borders for a clean look
            for row in range(4):
                axs[row][col].set_xticks([])
                axs[row][col].set_yticks([])
                axs[row][col].set_frame_on(False)

            # **Store figures for closing later (after saving overview)**
            figs_to_close.extend([fig_test, fig_train, fig_func_test, fig_func_train])
            # figs_to_close = [fig_test, fig_train, fig_func_test, fig_func_train]

            # Remove from the fifth as well
            axs_legend.set_xticks([])
            axs_legend.set_yticks([])
            axs_legend.set_frame_on(False)
            # # Add titles only to the first row
            # axs[0, col].set_title(f"{_model} - Test", fontsize=22, pad=15)

        # Define shift value (adjust as needed)
        shift = -0.3  # Move columns inward (reduce spacing)
        # Loop through all subplots and adjust left/right columns
        for row in range(4):
            for col in [0, 2]:  # Only adjust left (0) and right (2) columns
                pos = axs[row][col].get_position()  # Get current position
                x0, y0, width, height = pos.x0, pos.y0, pos.width, pos.height

                # Adjust left column (move right) & right column (move left)
                if col == 0:
                    axs[row][col].set_position([x0 + shift, y0, width, height])
                elif col == 2:
                    axs[row][col].set_position([x0 - shift, y0, width, height])

        # # Filter rule names where the probability (value) is > 0
        # rule_labels = [rule_name[r] for r in currentHP['rule_prob_map'].keys() if currentHP['rule_prob_map'][r] > 1e-10]
        # rule_shorts = [r for r in currentHP['rule_prob_map'].keys() if currentHP['rule_prob_map'][r] > 1e-10]
        # rule_handles = [plt.Line2D([0], [0], color=rule_color[r], lw=4) for r in rule_shorts]
        # Generate the legend image
        legend_img = create_legend_image()
        legend_ax = fig.add_axes([0.25, 0.0455, 0.25, 0.12])  # [left, bottom, width, height]
        legend_ax.imshow(legend_img)
        legend_ax.axis("off")  # Hide axis

        # Add Hyperparameters as Text Box
        hp_lines = [f"{key}: {currentHP[key]}" for key in selected_hp_keys if key in currentHP]
        num_columns = 3  # Adjust based on how wide the text should be
        hp_text_formatted = "\n".join(["   ".join(hp_lines[i:i + num_columns]) for i in range(0, len(hp_lines), num_columns)])

        # Position HP text BESIDE the legend (aligned inside the plot)
        axs_legend.text(0.515, 0.46, hp_text_formatted, transform=axs_legend.transAxes, fontsize=7,
                        verticalalignment='center', horizontalalignment='left', color='black',
                        bbox=dict(facecolor='white', alpha=1, edgecolor='black'))

        # Optimize layout for full visibility without stretching
        plt.tight_layout()
        # plt.show()
        plt.savefig(os.path.join(overviewFolder, _model.split("\\")[-1] + '_overview.png'), format='png', dpi=100, bbox_inches='tight')

        for f in figs_to_close:
            plt.close(f)

        plt.close(fig)

    except Exception as e:
        print(f"Error processing model {_model}: {e}")
        continue


