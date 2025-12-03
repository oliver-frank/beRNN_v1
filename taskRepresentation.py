# head: ################################################################################################################
# head: Analyze representation of tasks within hidden layers ###########################################################
# head: ################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import sys
import pickle
import seaborn as sns
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.linalg import orthogonal_procrustes
from scipy.stats import ttest_rel
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from pathlib import Path
from numpy import arctanh
from collections import OrderedDict

import tools
from network import Model
from analysis import clustering
from networkAnalysis import define_data_folder, rule_color
import matplotlib.patches as patches

# Functions ############################################################################################################
# Task Variance & Lesioning ############################################################################################
def plot_taskVariance_and_lesioning(directory, mode, sort_variable, rdm_metric, robustnessTest, batch, numberOfModels):
    # Colors used for clusters
    kelly_colors = \
        [np.array([0.94901961, 0.95294118, 0.95686275]),
         np.array([0.13333333, 0.13333333, 0.13333333]),
         np.array([0.95294118, 0.76470588, 0.]),
         np.array([0.52941176, 0.3372549, 0.57254902]),
         np.array([0.95294118, 0.51764706, 0.]),
         np.array([0.63137255, 0.79215686, 0.94509804]),
         np.array([0.74509804, 0., 0.19607843]),
         np.array([0.76078431, 0.69803922, 0.50196078]),
         np.array([0.51764706, 0.51764706, 0.50980392]),
         np.array([0., 0.53333333, 0.3372549]),
         np.array([0.90196078, 0.56078431, 0.6745098]),
         np.array([0., 0.40392157, 0.64705882]),
         np.array([0.97647059, 0.57647059, 0.4745098]),
         np.array([0.37647059, 0.30588235, 0.59215686]),
         np.array([0.96470588, 0.65098039, 0.]),
         np.array([0.70196078, 0.26666667, 0.42352941]),
         np.array([0.8627451, 0.82745098, 0.]),
         np.array([0.53333333, 0.17647059, 0.09019608]),
         np.array([0.55294118, 0.71372549, 0.]),
         np.array([0.39607843, 0.27058824, 0.13333333]),
         np.array([0.88627451, 0.34509804, 0.13333333]),
         np.array([0.16862745, 0.23921569, 0.14901961])]

    if robustnessTest == False:
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f'bestModels_{sort_variable}_{mode}.txt')
    else:
        txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', batch, f'bestModels_{sort_variable}_{mode}_{batch}.txt')

    with open(txtFile, "r") as file:
        lines = file.read().splitlines()
    cleaned_lines = [line.strip().strip('\'",') for line in lines]

    for model in range(1, numberOfModels + 1):
        best_model_dir = cleaned_lines[model]  # Choose model of interest, starting with [1]

        dataFolder = define_data_folder(best_model_dir.split('_'))

        participant = [i for i in best_model_dir.split('\\') if 'beRNN_' in i][0]
        data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)

        hp = tools.load_hp(best_model_dir)
        # tools.save_hp(hp_copy, best_model_dir) # pop out rng main reason for saving again
        # hp = tools.load_hp(best_model_dir)
        layer = [1 if hp['multiLayer'] == False else 3][0]

        # info: Create TaskVariance Plot
        knowledgeBase = clustering.Analysis(data_dir, best_model_dir, layer, rdm_metric, 'test', hp['monthsConsidered'],'rule', True)
        figurePath = os.path.join(directory, 'visuals')

        # Plot task variance anyway
        knowledgeBase.plot_variance(best_model_dir, os.path.join(directory, 'visuals'), mode_=f'{model}_{sort_variable}_{mode}')
        # But skip multiLayer
        if hp['multiLayer'] == True:
            continue
        else:
            knowledgeBase.plot_lesions(data_dir, best_model_dir, figurePath, mode_=f'{model}_{sort_variable}_{mode}')
########################################################################################################################

# Task Variance space ##################################################################################################
class TaskSetAnalysis(object):
    """Analyzing the representation of tasks."""
    def __init__(self, model_dir, rules=None):
        """Initialization.

        Args:
            model_dir: str, model directory
            rules: None or a list of rules
        """
        # Stimulus-averaged traces
        h_stimavg_byrule = OrderedDict()
        h_stimavg_byepoch = OrderedDict()
        # Last time points of epochs
        h_lastt_byepoch = OrderedDict()

        model = Model(model_dir)
        hp = model.hp

        if rules is None:  # Default value - all tasks
            rules = hp['rules']

        n_rules = len(rules)

        # Define main path
        path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'

        # Define data path
        preprocessedData_path = os.path.join(path, 'Data', hp['participant'], hp['data'])  # pandora

        with tf.Session() as sess:
            model.restore()

            for rule in rules:
                month = hp['monthsConsidered'][-1]
                train_data, test_data = tools.createSplittedDatasets(hp, preprocessedData_path, month)

                x, y, y_loc, response = tools.load_trials(hp['rng'], rule, 'test', hp['batch_size'], test_data,
                                                          False)  # y_loc is participantResponse_perfEvalForm
                c_mask = tools.create_cMask(y, response, hp, 'test')
                feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)

                h = sess.run(model.h, feed_dict=feed_dict)  # info: Trainables are actualized - train_step should represent the step in training.py and the global_step in network.py

                # c_lsq, c_reg, y_hat_test = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)

                # Average across trials
                h_stimavg = h.mean(axis=1)

                # dt_new = 50
                # every_t = int(dt_new/hp['dt'])

                t_start = int(500 / hp['dt'])  # Important: Ignore the initial transition
                # Extract epoch of interest - most often response epoch
                h_stimavg_byrule[rule] = h_stimavg[t_start:, :]

                fixation_steps, response_steps = tools.getEpochSteps(y)

                # Take epoch
                e_time_start = fixation_steps
                e_time_end = fixation_steps + response_steps
                e_time = [e_time_start, e_time_end]
                e_name = 'response'

                e_time_start = e_time[0] - 1 if e_time[0] > 0 else 0
                h_stimavg_byepoch[(rule, e_name)] = h_stimavg[e_time_start:e_time[1], :]
                # Take last time point from epoch
                # h_all_byepoch[(rule, e_name)] = np.mean(h[e_time[0]:e_time[1],:,:][-1], axis=1)
                h_lastt_byepoch[(rule, e_name)] = h[e_time[1] - 1, :, :]

        self.rules = rules
        self.h_stimavg_byrule = h_stimavg_byrule
        self.h_stimavg_byepoch = h_stimavg_byepoch
        self.h_lastt_byepoch = h_lastt_byepoch
        self.model_dir = model_dir

    @staticmethod  # utility function within class, no method
    def filter(h, rules=None, epochs=None, non_rules=None, non_epochs=None, get_lasttimepoint=True, get_timeaverage=False, **kwargs):
        # h should be a dictionary
        # get a new dictionary containing keys from the list of rules and epochs
        # And avoid epochs from non_rules and non_epochs
        # h_new = OrderedDict([(key, val) for key, val in h.items() if key[1] in epochs])

        if get_lasttimepoint:
            print('Analyzing last time points of epochs')
        if get_timeaverage:
            print('Analyzing time-averaged activities of epochs')

        h_new = OrderedDict()
        for key in h:
            rule, epoch = key

            include_key = True
            if rules is not None:
                include_key = include_key and (rule in rules)

            if epochs is not None:
                include_key = include_key and (epoch in epochs)

            if non_rules is not None:
                include_key = include_key and (rule not in non_rules)

            if non_epochs is not None:
                include_key = include_key and (epoch not in non_epochs)

            if include_key:
                if get_lasttimepoint:
                    h_new[key] = h[key][np.newaxis, -1, :]
                elif get_timeaverage:
                    h_new[key] = np.mean(h[key], axis=0, keepdims=True)
                else:
                    h_new[key] = h[key]

        return h_new

    def compute_taskspace(self, fname, dim_reduction_type, epochs, **kwargs):
        # Only get last time points for each epoch
        h = self.filter(self.h_stimavg_byepoch, epochs=epochs, rules=self.rules, **kwargs)

        # Concatenate across rules to create dataset
        data = np.concatenate(list(h.values()), axis=0)
        data = data.astype(dtype='float64')

        # First reduce dimension to dimension of data points
        from sklearn.decomposition import PCA
        n_comp = int(np.min([data.shape[0], data.shape[1]]) - 1)
        model = PCA(n_components=n_comp)
        data = model.fit_transform(data)

        if dim_reduction_type == 'PCA':
            model = PCA(n_components=2)

        elif dim_reduction_type == 'MDS':
            from sklearn.manifold import MDS
            model = MDS(n_components=2, metric=True, random_state=0)

        elif dim_reduction_type == 'TSNE':
            from sklearn.manifold import TSNE
            model = TSNE(n_components=2, init='pca',
                         verbose=1, method='exact', learning_rate=100, perplexity=5)

        elif dim_reduction_type == 'IsoMap':
            from sklearn.manifold import Isomap
            model = Isomap(n_components=2)

        else:
            raise ValueError('Unknown dim_reduction_type')

        # Transform data
        data_trans = model.fit_transform(data)

        # Package back to dictionary
        h_trans = OrderedDict()
        i_start = 0
        for key, val in h.items():
            i_end = i_start + val.shape[0]
            h_trans[key] = data_trans[i_start:i_end, :]
            i_start = i_end

        # save file
        with open(fname, "wb") as f:
            pickle.dump(h_trans, f)

        return h_trans

    def plot_taskspace(self, h_trans, directory, sort_variable, mode, fig_name, level, plot_example=False, lxy=None, plot_arrow=True, **kwargs):
        figsize = (5, 5)
        fs = 7  # fontsize
        dim0, dim1 = (0, 1)  # plot dimensions
        i_example = 0  # index of the example to plot

        texts = list()

        maxv0, maxv1 = -1, -1

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0.2, 0.2, 0.65, 0.65])

        for key, val in h_trans.items():
            rule, epoch = key
            # Default coloring by rule_color
            color = rule_color[rule]

            if plot_example:
                xplot, yplot = val[i_example, dim0], val[i_example, dim1]
            else:
                xplot, yplot = val[:, dim0], val[:, dim1]

            # ax.plot(xplot, yplot, 'o', color=color, mec=color, mew=1.0, ms=2)
            ax.plot(xplot, yplot, 'o', color=color, mec=color, mew=0.5, ms=5, alpha=0.4)

            xtext = np.mean(val[:, dim0])
            if np.mean(val[:, dim1]) > 0:
                ytext = np.max(val[:, dim1])
                va = 'bottom'
            else:
                ytext = np.min(val[:, dim1])
                va = 'top'

            texts.append(ax.text(xtext * 1.1, ytext * 1.1, tools.rule_name[rule],
                                 fontsize=6, color=color, alpha=0.6,
                                 horizontalalignment='center', verticalalignment=va))

            maxv0 = np.max([maxv0, np.max(abs(val[:, dim0]))])
            maxv1 = np.max([maxv1, np.max(abs(val[:, dim1]))])

        if lxy is None:
            lx = np.ceil(maxv0)
            ly = np.ceil(maxv1)
        else:
            lx, ly = lxy

        ax.tick_params(axis='both', which='major', labelsize=fs)
        # plt.locator_params(nbins=3)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.margins(0.1)
        # plt.axis('equal')
        plt.xlim([-lx, lx])
        plt.ylim([-ly, ly])
        ax.plot([0, 0], [-ly, ly], '--', color='gray')
        ax.plot([-lx, lx], [0, 0], '--', color='gray')
        ax.set_xticks([-lx, lx])
        ax.set_yticks([-ly, ly])
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        pc_name = 'rPC'
        ax.set_xlabel(pc_name + ' {:d}'.format(dim0 + 1), fontsize=fs, labelpad=-5)
        ax.set_ylabel(pc_name + ' {:d}'.format(dim1 + 1), fontsize=fs, labelpad=-5)

        finalDirectory = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'compositionalRepresentation')
        os.makedirs(finalDirectory, exist_ok=True)
        if level == 'individual':
            plt.savefig(os.path.join(finalDirectory, fig_name + '.png'), transparent=True, dpi=300)
        elif level == 'group':
            plt.savefig(os.path.join(finalDirectory, '2DtaskRepresentation_overview.png'), transparent=True, dpi=300)
        # plt.show()

    def collect_h_trans(self, h_trans, h_trans_all, model):
        # Collect h_trans values
        h_trans_values = list(h_trans.values())

        # rotation_matrix, clock wise
        get_angle = lambda vec: np.arctan2(vec[1], vec[0])
        theta = get_angle(h_trans_values[0][0])
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])

        for key, val in h_trans.items():
            val_rot = np.dot(val, rot_mat)
            if key in h_trans_all:
                h_trans_all[key] = np.concatenate((h_trans_all[key], val_rot), axis=0)
            else:
                h_trans_all[key] = val_rot

        h_trans_values = list(h_trans_all.values())
        if h_trans_values[1][0][1] < 0:
            for key, val in h_trans_all.items():
                h_trans_all[key] = val * np.array([1, -1])

        return h_trans_all

########################################################################################################################

# RDM & RSA ############################################################################################################
def plot_group_rdm_mds(directory, mode, sort_variable, rdm_metric, numberOfModels, ruleset):
    def plot_rdm_heatmap(rdm, metric, task_labels=None, title='RDM Heatmap'):
        n_tasks = rdm.shape[0]
        fig_size = max(6, n_tasks * 0.5)  # auto-scale figure for more tasks

        plt.figure(figsize=(fig_size, fig_size * 0.85))
        ax = sns.heatmap(rdm, annot=False, cmap='viridis', square=True,
                         xticklabels=task_labels, yticklabels=task_labels,
                         cbar_kws={'shrink': 0.7})

        plt.title(title + ' - ' + metric, fontsize=12)
        plt.xlabel('Tasks', fontsize=10)
        plt.ylabel('Tasks', fontsize=10)

        # Rotate x-axis labels and adjust font sizes
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)

        plt.tight_layout()
        # plt.show()

        return plt

    def plot_rdm_mds(rdm, metric, task_keys, rule_color, title='RDM MDS'):
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
        coords = mds.fit_transform(rdm)

        plt.figure(figsize=(5.2, 4.5))
        ax = plt.gca()

        for i, key in enumerate(task_keys):
            x, y = coords[i]
            color = rule_color.get(key, 'gray')
            label = key

            # Dot
            ax.scatter(x, y, color=color, s=60, edgecolor='black', zorder=3)

            # Label outside the dot, same color
            ax.text(x + 0.035, y, label, fontsize=6.5,
                    color=color, ha='left', va='center', zorder=4)

        # Coordinate system (axes on, ticks small and clean)
        ax.tick_params(axis='both', which='major', labelsize=6, length=2)
        ax.set_xlabel('MDS Dimension 1', fontsize=7.5)
        ax.set_ylabel('MDS Dimension 2', fontsize=7.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')

        plt.title(title + ' - ' + metric, fontsize=9)
        plt.tight_layout()
        # plt.show()

    all_coords = []
    all_keys = []
    all_models = []
    # Apply non-linear dimensionality reduction that maps element-wise distance as good as possible into n-dim space
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)  # euclidean distance based by default

    for model in range(1, numberOfModels+1):  # Best of x models - max. 256 - always start with 1
        best_model_dir = cleaned_lines[model]

        if best_model_dir == '':
            print('No best model saved in text file.')
            exit()

        dataFolder = define_data_folder(best_model_dir.split('_'))

        participant = [i for i in best_model_dir.split('\\') if 'beRNN_' in i and len(i) == 8][0]

        data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)

        hp = tools.load_hp(best_model_dir)
        layer = [1 if hp['multiLayer'] == False else 3][0]

        # Create task variance matrix for current model in loop
        knowledgeBase = clustering.Analysis(data_dir, best_model_dir, layer, rdm_metric, 'test', hp['monthsConsidered'],'rule', True)

        # # Skip dummy RDMs
        # if np.allclose(knowledgeBase.rdm, knowledgeBase.rdm[0, 0]):
        #     print(f"Skipping model {model} from final plot due to dummy RDM (constant dissimilarity).")
        #     continue

        if model == 1:
            coords_ref = mds.fit_transform(knowledgeBase.rdm)
            coords_aligned = coords_ref
        else:
            coords_model = mds.fit_transform(knowledgeBase.rdm)
            R, _ = orthogonal_procrustes(coords_model, coords_ref)
            coords_aligned = coords_model @ R

        all_coords.append(coords_aligned)
        all_keys.append(knowledgeBase.keys)
        all_models.append(knowledgeBase.model_dir)

        # # info: Plot also single performance plots for each of the chosen models
        # rule_plot = [i for i in hp['rule_prob_map'] if hp['rule_prob_map'][i] > 0]
        # performanceTest_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'performanceTest')
        # performanceTrain_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'performanceTrain')
        if robustnessTest == True:
            representationalDissimilarity_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', str(batch),
                                                             f'representationalDissimilarity_{rdm_metric}_{ruleset}')
        else:
            representationalDissimilarity_dir = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}',
                                                             f'representationalDissimilarity_{rdm_metric}_{ruleset}')

        # os.makedirs(performanceTest_dir, exist_ok=True)
        # os.makedirs(performanceTrain_dir, exist_ok=True)
        os.makedirs(representationalDissimilarity_dir, exist_ok=True)

        # save rdm
        np.save(os.path.join(representationalDissimilarity_dir,
                             f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[-5] + '_' +
                             best_model_dir.split("\\")[-3].split('_')[
                                 -4] + f'_rdmArray_{knowledgeBase.rdm_metric}_{ruleset}.npy'), knowledgeBase.rdm)

        # info: Create RDM Heatmaps and 2D representations
        label_list = [tools.rule_name[key] for key in knowledgeBase.keys]
        fig_rdm = plot_rdm_heatmap(knowledgeBase.rdm, knowledgeBase.rdm_metric, task_labels=label_list)
        fig_rdm.savefig(os.path.join(representationalDissimilarity_dir,
                                     f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[
                                         -5] + '_' + best_model_dir.split("\\")[-3].split('_')[
                                         -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}.png'),
                        format='png', dpi=300)
        fig_rdm_mds = plot_rdm_mds(knowledgeBase.rdm, knowledgeBase.rdm_metric, task_keys=knowledgeBase.keys,
                                   rule_color=rule_color)
        fig_rdm.savefig(os.path.join(representationalDissimilarity_dir,
                                     f'{model}_{sort_variable}_{mode}' + '_' + 'batch_' + best_model_dir.split("\\")[
                                         -5] + '_' + best_model_dir.split("\\")[-3].split('_')[
                                         -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}_2DspaceGeometry.png'),
                        format='png', dpi=300)

    if numberOfModels > 1:
        plt.figure(figsize=(5.2, 4.5))
        ax = plt.gca()

        # Track which keys have already been labeled
        labeled_keys = set()

        for coords, keys in zip(all_coords, all_keys):
            for x, y, key in zip(coords[:, 0], coords[:, 1], keys):
                color = rule_color.get(key, 'gray')
                # Only set label for the first occurrence
                label = key if key not in labeled_keys else None

                ax.scatter(x, y, color=color, s=60, edgecolor='black', zorder=3, alpha=0.6)
                ax.text(x + 0.035, y, label, fontsize=6.5,
                        color=color, ha='left', va='center', zorder=4)

        # Only show unique legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=6.5, loc='best')

        ax.tick_params(axis='both', which='major', labelsize=6, length=2)
        ax.set_xlabel('MDS Dimension 1', fontsize=7.5)
        ax.set_ylabel('MDS Dimension 2', fontsize=7.5)
        ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        plt.title(f'RDM MDS – {knowledgeBase.rdm_metric} – Multiple Aligned Models', fontsize=9)
        plt.tight_layout()
        # plt.show()

        plt.savefig(os.path.join(representationalDissimilarity_dir, best_model_dir.split("\\")[-3].split('_')[
            -4] + f'_representationalDissimilarity_{knowledgeBase.rdm_metric}_{ruleset}_2DspaceGeometry_modelAlignment_{numberOfModels}.png'), format='png', dpi=300)
        plt.close()

def plot_rsa(directory, participantList):
    def ascendingNumbers(e):
        return int(e.split('_')[0])
    def vec(rdm):  # → length 66 (for 12 tasks)
        idx = np.triu_indices_from(rdm, k=1)
        return rdm[idx]

    rsa_directory = Path(*directory.parts[:-1], '_networkComparison')
    os.makedirs(rsa_directory, exist_ok=True)

    # Gather rdmFiles in dict lists
    rdm_dict = {}
    for participant in participantList:

        # attention: *****************************
        if participant == 'beRNN_03':
            directory = Path(f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/_robustnessTest_multiTask_{participant}_highDimCorrects_256/{dataType}/{participant}')
            robustnessTest = True
            batch = 4
        else:
            directory = Path(
                f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/_robustnessTest_multiTask_{participant}_highDimCorrects_256_hp_4/{dataType}/{participant}')
            robustnessTest = False
        # attention: *****************************

        # info: different foldername endings for representationalDissimilarity_cosine possible - independent of naming, the files within will be correct
        if robustnessTest == True:
            directory_ = Path(directory, f'visuals/performance_test/batchPlots/{str(batch)}', [directory for directory in os.listdir(Path(*directory.parts[:-1], f'{participant}/visuals/performance_test/batchPlots/{str(batch)}')) if 'representationalDissimilarity_cosine' in directory][0])
        else:
            directory_ = Path(directory, 'visuals/performance_test', [directory for directory in os.listdir(Path(*directory.parts[:-1], f'{participant}/visuals/performance_test')) if 'representationalDissimilarity_cosine' in directory][0])

        # Check if all defined participant in list were preprocessed - exit function if not
        if not directory_.exists():
            print(f"Directory {directory_} does not exist. Exiting.")
            sys.exit()

        rdmFiles = [i for i in os.listdir(str(directory_).format(participant=participant)) if i.endswith('.npy')]
        rdmFiles.sort(key=ascendingNumbers)  # Sort list according to information chunk given in key function
        rdm_dict[participant] = rdmFiles

    # Align rdmLists to same length
    min_length = min([len(rdm_dict[rdmList]) for rdmList in rdm_dict])
    # min_length = 10
    for participant in participantList:
        rdm_dict[participant] = rdm_dict[participant][:min_length]
    # Load the rdm files
    for participant in participantList:

        # attention: *****************************
        if participant == 'beRNN_03':
            directory = Path(
                f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/_robustnessTest_multiTask_{participant}_highDimCorrects_256/{dataType}/{participant}')
            robustnessTest = True
            batch = 4
        else:
            directory = Path(
                f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/_robustnessTest_multiTask_{participant}_highDimCorrects_256_hp_4/{dataType}/{participant}')
            robustnessTest = False
        # # attention: *****************************

        # info: different foldername endings for representationalDissimilarity_cosine possible - independent of naming, the files within will be correct
        if robustnessTest == True:
            directory_ = Path(directory, f'visuals/performance_test/batchPlots/{str(batch)}', [directory for directory in os.listdir(Path(*directory.parts[:-1], f'{participant}/visuals/performance_test/batchPlots/{str(batch)}')) if 'representationalDissimilarity_cosine' in directory][0])
        else:
            directory_ = Path(directory, 'visuals/performance_test', [directory for directory in os.listdir(Path(*directory.parts[:-1], f'{participant}/visuals/performance_test')) if 'representationalDissimilarity_cosine' in directory][0])


        for rdm in range(min_length):
                rdm_dict[participant][rdm] = np.load(os.path.join(str(directory_).format(participant=participant), rdm_dict[participant][rdm]))

    # Create vectors from rdm ndarrays for upper triangle of symmetric rdms
    rdm_vec_dict = {s: [vec(r) for r in rdm_list] for s, rdm_list in rdm_dict.items()}

    # Compute spearman rank correlation for each possible combination of model groups (within/between)
    subjects = list(rdm_vec_dict.keys())
    N_subj = len(subjects)
    within_rsa = {s: [] for s in subjects}
    between_rsa = {s: [] for s in subjects}

    # Loop over all combinations and assign results to the right dict (within/between)
    for i, s1 in enumerate(subjects):
        for j, s2 in enumerate(subjects):
            for v1 in rdm_vec_dict[s1]:
                for v2 in rdm_vec_dict[s2]:
                    rho = scipy.stats.spearmanr(v1, v2).correlation
                    if s1 == s2:
                        within_rsa[s1].append(rho)
                    else:
                        between_rsa[s1].append(rho)

    # Create dict for each comparison's mean value
    within_mean = np.array([np.mean(within_rsa[s]) for s in subjects])  # shape (5,)
    between_mean = np.array([np.mean(between_rsa[s]) for s in subjects])  # shape (5,)

    z_within = arctanh(within_mean)  # Fisher z
    z_between = arctanh(between_mean)

    t, p = ttest_rel(z_within, z_between)  # H₀: means equal
    print(f'Within  mean ρ = {within_mean.mean():.3f}')
    print(f'Between mean ρ = {between_mean.mean():.3f}')
    print(f'Paired t(4) = {t:.2f}, p = {p:.4f}')

    # Create ordered list of vectors and labels
    all_vecs = []
    labels = []
    for subj in subjects:
        for i, vec in enumerate(rdm_vec_dict[subj], 1):
            all_vecs.append(vec)
            labels.append(f"{subj}_M{i}")

    # Compute full Spearman RSA similarity matrix
    n_models = len(all_vecs)
    rsa_sim = np.zeros((n_models, n_models))
    for i in range(n_models):
        for j in range(n_models):
            rho, _ = spearmanr(all_vecs[i], all_vecs[j])
            rsa_sim[i, j] = rho

    # Convert to dissimilarity
    rsa_dissim = 1 - rsa_sim
    print('maxValue', np.max(rsa_dissim))
    # Plot RSA heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        rsa_dissim,
        cmap="viridis",
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        linewidths=0,
        cbar_kws={
            'shrink': 0.5,
            'aspect': 10,
            'label': 'Representational Dissimilarity (1 - Spearman ρ)',
            'ticks': np.linspace(0, 0.5, 6)
        },
        vmin=0, vmax=0.5
    )

    # ----- SET CUSTOM TICKS -----
    n_groups = len(subjects)
    group_size = 20  # number of models per participant
    tick_positions = [i * group_size + group_size // 2 for i in range(n_groups)]

    # Draw rectangles around diagonal blocks
    for g in range(n_groups):
        start = g * group_size
        rect = patches.Rectangle(
            (start, start),  # (x, y) start position
            group_size,  # width
            group_size,  # height
            fill=False,  # no fill
            edgecolor='lightsteelblue',  # border color
            linewidth=3  # line thickness
        )
        ax.add_patch(rect)

    # Create text lines
    stats_text = (
        f"Within mean ρ = {within_mean.mean():.3f}\n"
        f"Between mean ρ = {between_mean.mean():.3f}\n"
        f"Paired t(4) = {t:.2f}, p = {p:.4f}"
    )

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # Colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('RDA (1 - Spearman ρ)', fontsize=12)

    ax.set_xticklabels(subjects, rotation=0, fontsize=12)
    ax.set_yticklabels(subjects, rotation=90, fontsize=12)

    plt.title("RDA matrix", fontsize=20, fontweight='bold')
    plt.xlabel("Participant", fontsize=16)
    plt.ylabel("Participant", fontsize=16)
    plt.tight_layout()

    plt.subplots_adjust(bottom=0.2)  # space for stats

    plt.figtext(
        0.435, 0.15,
        stats_text,
        ha='center', va='top',
        fontsize=14,
        linespacing=1.5
    )

    plt.show()
    plt.savefig(os.path.join(rsa_directory, f'RDAmatrix_h_normMean_all_{within_mean.mean():.3f}.png'))
########################################################################################################################

# Task representation analysis - variable allocation ###################################################################
# info: The script can only be run after participants have been analyzed by hyperparameterOverview.py
mode = ['test', 'train'][0]
sort_variable = ['performance', 'clustering'][0]
rdm_metric = ['cosine', 'correlation'][0]
representation = ['rate', 'weight'][0]
restore = False

participantList =  ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
folders = ['_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_2', '_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_7',
           '_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_4', '_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_10',
           '_robustnessTest_multiTask_beRNN_05_highDimCorrects_256_hp_5']
standard_analysis = [True, False][0]
rsa_analysis = [True, False][1]
robustnessTest, batch = [True, False][1], '2'
numberOfModels = 20 # max. number of models in folder

for folder in folders:

    dataType = 'highDim_correctOnly' if 'highDim_correctOnly' in folder or 'highDimCorrects' in folder else 'highDim'
    participant = [participant for participant in participantList if participant in folder][0]
    directory = Path(f'C:/Users/oliver.frank/Desktop/PyProjects/beRNNmodels/{folder}/{dataType}/{participant}')

    if standard_analysis == True:

        # Choose file naming accoridng to represented tasks
        if 'fundamentals' in folder:
            ruleset = 'fundamentals'
        elif 'multiTask' in folder:
            ruleset = 'all'
        else:
            ruleset = 'domainTask'

        # Define for different folder structure
        if robustnessTest == False:
            txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}',
                                   f'bestModels_{sort_variable}_{mode}.txt')
        else:
            txtFile = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', batch,
                                   f'bestModels_{sort_variable}_{mode}_{batch}.txt')

        with open(txtFile, "r") as file:
            lines = file.read().splitlines()
        cleaned_lines = [line.strip().strip('\'",') for line in lines]


        # head: Create individual task variance and lesioning plots ##############################################################
        # plot_taskVariance_and_lesioning(directory, mode, sort_variable, rdm_metric, robustnessTest, batch, numberOfModels=numberOfModels)


        # head: Create task variance space plots - tsne/PCA based ################################################################
        # tsa_exist = False
        # h_trans_all = OrderedDict()
        # # Iterate over all models and create TR individually and on group-level
        # for model in range(1, numberOfModels + 1):
        #     # rules = tools.rules_dict[ruleset]
        #
        #     best_model_dir = cleaned_lines[model]  # Choose model of interest, starting with [1]
        #
        #     fname = 'taskset{:s}_space_2DtaskVarianceRepresentation'.format(ruleset) + '.pkl'  # fix: set subgroups of tasks as variable here too
        #     fname = os.path.join(best_model_dir, fname)
        #     figName = 'taskset{:s}_space_2DtaskVarianceRepresentation_'.format(ruleset) + '_'.join(best_model_dir.split('\\')[-3].split('_')[-5:-3])
        #
        #     try:
        #         tsa, tsa_exist = TaskSetAnalysis(best_model_dir), True
        #         h_trans = tsa.compute_taskspace(fname, dim_reduction_type='MDS', epochs=['response']) # Focus on response epoch
        #         tsa.plot_taskspace(h_trans, directory, sort_variable, mode, figName, 'individual')
        #         # Collect all h_trans (2DtaskRepresentations) for group plot
        #         h_trans_all = tsa.collect_h_trans(h_trans, h_trans_all, model)
        #     except ValueError:
        #         print('Skipping model: ' + best_model_dir.split('\\')[-3])
        #         continue
        #
        # if tsa_exist:
        #     tsa.plot_taskspace(h_trans_all, directory, sort_variable, mode, figName, 'group', lxy=(1.1, 1.1))


        # head: Create individual rdm heatmaps, rdm 2D representations and one grouped rdm 2D representation #####################
        plot_group_rdm_mds(directory, mode, sort_variable, rdm_metric, numberOfModels, ruleset)


    # Create RSA matrix for comparing rdm representations between participants #############################################
    if rsa_analysis == True:
        # participantList = ['beRNN_03']
        plot_rsa(directory, participantList)


