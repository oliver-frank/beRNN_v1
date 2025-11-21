########################################################################################################################
# head: hp overview ####################################################################################################
########################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import defaultdict
from collections import OrderedDict
import os
import json
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import tools
# from analysis import variance
from analysis import clustering
from networkAnalysis import define_data_folder
# from analysis import standard_analysis

########################################################################################################################
# head: Create histogramms to visualize and investigate interrelations of hyperparameter, modularity and performance ###
########################################################################################################################
def compute_n_cluster(model_dirs, mode):
    successful_model_dirs = []

    for model_dir in model_dirs:
        print(model_dir)
        try:
            log = tools.load_log(model_dir)
            # log = tools.load_log(r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\paperPlanes\highDim3stimTC\beRNN_04\13\beRNN_04_AllTask_4-6_data_highDim_correctOnly_3stimTC_trainingBatch13_iteration6_LeakyGRU_128_softplus\model_month_6')
            hp = tools.load_hp(model_dir)
            # info: Add try, except and assert if you only want to take models into account that overcome certain performance threshold
            dataFolder = define_data_folder(model_dir.split('_'))
            # participant = [i for i in model_dir.split('\\') if 'beRNN_' in i][0]
            participant = '_'.join(['beRNN', [string for string in model_dir.split('_') if '0' in string and len(string) == 2][0]]) # fix new ---
            layer = [1 if hp['multiLayer'] == False else 3][0]

            # Info: Important overwriting of incongruent information in hp between single and multiLayer architecture
            hp['n_rnn'] = hp['n_rnn_per_layer'][0] if hp.get('multiLayer') else hp['n_rnn']
            hp['activation'] = hp['activations_per_layer'][0] if hp.get('multiLayer') else hp['activation']
            tools.save_hp(hp, model_dir)

            # Define right data
            data_dir = os.path.join('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\Data', participant, dataFolder)
            rdm_metric = 'cosine'
            if mode == 'test':
                analysis = clustering.Analysis(data_dir, model_dir, layer, rdm_metric,'test', hp['monthsConsidered'], 'rule', True) # test performance
            elif mode == 'train':
                analysis = clustering.Analysis(data_dir, model_dir, layer, rdm_metric,'train', hp['monthsConsidered'], 'rule', True) # train performance

            # Average performance for training at the last time point
            totalPerformanceTraining = 0
            totalPerformanceTesting = 0
            numberOfTasks = 0

            tasksToTakeIntoAccount = [i for i in hp['rule_prob_map'] if hp['rule_prob_map'][i] > 0]

            for key in log.keys():
                if 'perf_train' in key and 'avg' not in key and any(task in key for task in tasksToTakeIntoAccount): # Only rule_prob_map > 0 are saved during training
                    totalPerformanceTraining += log[key][-1]
                    numberOfTasks += 1
            averageTotalPerformanceTraining = totalPerformanceTraining / numberOfTasks

            for key in log.keys():
                if 'perf_' in key and 'avg' not in key and any(task for task in tasksToTakeIntoAccount if task == key.split('perf_')[-1]): # All tasks in rule_prob_map are saved during training
                    totalPerformanceTesting += log[key][-1]
            averageTotalPerformanceTesting = totalPerformanceTesting / numberOfTasks

            log['avg_perf_train'] = averageTotalPerformanceTraining
            log['avg_perf_test'] = averageTotalPerformanceTesting
            # log['avg_perf_test'] = log['perf_avg'][-1]
            log['n_cluster'] = analysis.n_cluster
            log['score'] = max(analysis.scores)
            log['model_dir'] = model_dir
            tools.save_log(log)

            # except IOError:
                # Training never finished
                # assert log['perf_min'][-1] <= hp['target_perf']

            # analysis.plot_example_unit()
            # analysis.plot_variance()
            # analysis.plot_2Dvisualization()

            successful_model_dirs.append(model_dir)
            print("done")

        except Exception as e:
            print(f"An exception occurred in compute_n_cluster: {e}")

            # Create dummy log for plotting
            log = {}

            log['avg_perf_train'] = 0
            log['avg_perf_test'] = 0
            log['n_cluster'] = 0
            log['score'] = 0
            log['model_dir'] = model_dir

            tools.save_log(log)

            successful_model_dirs.append(model_dir)
            print("fallback done - dummy log created")

    return successful_model_dirs

def get_n_clusters(model_dirs):
    # model_dirs = tools.valid_model_dirs(root_dir)
    hp_list = list()
    n_clusters = list()
    silhouette_score = list()
    avg_perf_train_list = list()
    avg_perf_test_list = list()
    modularity_list_sparse = list()

    for i, model_dir in enumerate(model_dirs):
        if i % 50 == 0:
            print('Analyzing model {:d}/{:d}'.format(i, len(model_dirs)))
        print(model_dir)
        hp = tools.load_hp(model_dir)
        log = tools.load_log(model_dir)

        # Handle internal matplotlib issue with None values for plotting legend
        if hp['learning_rate_mode'] is None:
            hp['learning_rate_mode'] = 'constant'
            print('None overwritten with "constant"')
        hp['rnn_type'] = 'MultiLayer' if hp.get('multiLayer') else hp['rnn_type']
        # load hp and calculate acerage performance
        # if average performance > threshold

        # check if performance exceeds target
        # if log['perf_min'][-1] > hp['target_perf']: # fix
        n_clusters.append(log['n_cluster'])
        hp_list.append(hp)
        silhouette_score.append(log['score'])
        avg_perf_train_list.append(log['avg_perf_train'])
        avg_perf_test_list.append(log['avg_perf_test'])
        if hp.get('multiLayer') == False:
            try:
                modularity_list_sparse.append(log['modularity_sparse'][-1])
            except Exception as e:
                modularity_list_sparse.append(0)

        else:
            modularity_list_sparse = []

    return n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse

def plot_histogram():
    initdict = defaultdict(list)
    initdictother = defaultdict(list)
    initdictotherother = defaultdict(list)

    for model_dir in model_dirs:
        hp = tools.load_hp(model_dir)
        # check if performance exceeds target
        log = tools.load_log(model_dir)
        # if log['perf_avg'][-1] > hp['target_perf']:
        if log['perf_min'][-1] > hp['target_perf']: # attention: define well - maybe conservative th: 0.5 & liberal th: 0.7
            print('no. of clusters', log['n_cluster'])
            n_clusters.append(log['n_cluster'])
            hp_list.append(hp)

            initdict[hp['w_rec_init']].append(log['n_cluster'])
            initdict[hp['activation']].append(log['n_cluster'])

            # initdict[hp['rnn_type']].append(log['n_cluster'])
            # if hp['activation'] != 'tanh': fix: why no tanh??
            initdict[hp['rnn_type']].append(log['n_cluster'])
            initdictother[hp['rnn_type'] + hp['activation']].append(log['n_cluster'])
            initdictotherother[hp['rnn_type'] + hp['activation'] + hp['w_rec_init']].append(log['n_cluster'])

            if hp['l1_h'] == 0:
                initdict['l1_h_0'].append(log['n_cluster'])
            else:  # hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:
                keyvalstr = 'l1_h_1emin' + str(int(abs(np.log10(hp['l1_h']))))
                initdict[keyvalstr].append(log['n_cluster'])

            # fix: l1 only good?
            if hp['l1_weight'] == 0:
                initdict['l1_weight_0'].append(log['n_cluster'])
            else:  # hp['l1_h'] == 1e-3 or 1e-4 or 1e-5:
                keyvalstr = 'l1_weight_1emin' + str(int(abs(np.log10(hp['l1_weight']))))
                initdict[keyvalstr].append(log['n_cluster'])

                # initdict[hp['l1_weight']].append(log['n_cluster'])

    # Check no of clusters under various conditions.
    f, axarr = plt.subplots(7, 1, figsize=(3, 12), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_' not in key:
            title = (key + ' ' + str(len(initdict[key])) +
                     ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].set_title(title)
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_96nets.png')
    # plt.savefig('./figure/histforcases__pt9_192nets.pdf')
    # plt.savefig('./figure/histforcases___leakygrunotanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdictother.keys():
        if 'l1_' not in key:
            axarr[u].set_title(
                key + ' ' + str(len(initdictother[key])) + ' mean: ' + str(round(np.mean(initdictother[key]), 2)))
            axarr[u].hist(initdictother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases__leakyrnngrurelusoftplus_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 6), sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'diag' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key])) + ' mean: ' + str(
                round(np.mean(initdictotherother[key]), 2)))
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_randortho_notanh_pt9_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 6), sharex=True)
    u = 0
    for key in initdictotherother.keys():
        if 'l1_' not in key and 'randortho' not in key:
            axarr[u].set_title(key + ' ' + str(len(initdictotherother[key])) + ' mean: ' + str(
                round(np.mean(initdictotherother[key]), 2)))
            axarr[u].hist(initdictotherother[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/histforcases_diag_notanh_pt9_192nets.pdf')

    # regu--
    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_h_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key])) + ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/noofclusters_pt9_l1_h_192nets.pdf')

    f, axarr = plt.subplots(4, 1, figsize=(3, 8), sharex=True)
    u = 0
    for key in initdict.keys():
        if 'l1_weight_' in key:
            axarr[u].set_title(key + ' ' + str(len(initdict[key])) + ' mean: ' + str(round(np.mean(initdict[key]), 2)))
            axarr[u].hist(initdict[key])
            u += 1
    f.subplots_adjust(wspace=.3, hspace=0.3)
    # plt.savefig('./figure/noofclusters_pt9_l1_weight_192nets.pdf')

def _get_hp_ranges():
    """Get ranges of hp."""
    hp_ranges = OrderedDict()
    hp_ranges['activation'] = ['softplus', 'relu', 'tanh']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU', 'MultiLayer']
    hp_ranges['n_rnn'] = [8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
    hp_ranges['w_rec_init'] = ['randortho', 'randgauss', 'diag', 'brainStructure']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_weight'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['learning_rate'] = [0.002, 0.0015, 0.001, 0.0005, 0.0001, 0.00005]
    hp_ranges['learning_rate_mode'] = ['constant', 'exp_range', 'triangular2']
    # hp_ranges['errorBalancingValue'] = [1., 5.]
    return hp_ranges

def general_hp_plot(n_clusters, silhouette_score, hp_list, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse, directory, sort_variable, mode, batchPlot, model_dir_batches):
    hp_ranges = _get_hp_ranges()
    hp_plots = list(hp_ranges.keys())

    # Sort by descending number of sort_variable
    if sort_variable == 'performance' and mode == 'test':
        ind_sort = np.argsort(avg_perf_test_list)[::-1]
    elif sort_variable == 'performance' and mode == 'train':
        ind_sort = np.argsort(avg_perf_train_list)[::-1]
    elif sort_variable == 'clustering':
        ind_sort = np.argsort(n_clusters)[::-1]
    elif sort_variable == 'silhouette':
        ind_sort = np.argsort(silhouette_score)[::-1]
    elif sort_variable == 'modularity':
        ind_sort = np.argsort(modularity_list_sparse)[::-1]

    n_clusters_sorted = [n_clusters[i] for i in ind_sort]
    silhouette_score_sorted = [silhouette_score[i] for i in ind_sort]
    hp_list_sorted = [hp_list[i] for i in ind_sort]
    avg_perf_train_list_sorted = [avg_perf_train_list[i] for i in ind_sort]
    avg_perf_test_list_sorted = [avg_perf_test_list[i] for i in ind_sort]
    if hp_list[0]['rnn_type'] != 'MultiLayer':
        modularity_list_sparse_sorted = [modularity_list_sparse[i] for i in ind_sort]
    successful_model_dirs_sorted = [successful_model_dirs[i] for i in ind_sort]

    # Prepare heatmap data
    hp_visualize = np.zeros((len(hp_plots), len(n_clusters)))
    color_indices_per_hp = {}
    for i, hp_name in enumerate(hp_plots):
        values = [hp[hp_name] for hp in hp_list_sorted]
        unique_vals = hp_ranges[hp_name]
        color_indices_per_hp[hp_name] = {v: j / (len(unique_vals) - 1) for j, v in enumerate(unique_vals)}
        for j, val in enumerate(values):
            hp_visualize[i, j] = color_indices_per_hp[hp_name][val]

    # === MAIN PLOTS ===
    fig, axs = plt.subplots(5, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.5]})
    plt.subplots_adjust(hspace=0.5)

    if hp_list[0]['rnn_type'] != 'MultiLayer': # attention: You have to add the calculation for modularity for multiRNN
        axs[0].plot(modularity_list_sparse_sorted, '-')
        axs[0].set_ylabel(f'Modularity score ({mode})', fontsize=7)
        axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)
        # Add light grey dashed lines at y=0.3, 0.5, 0.7
        for y in [0.3, 0.5, 0.7]:
            axs[0].axhline(y=y, color='lightgrey', linestyle='--', linewidth=0.8, zorder=0)
    else:
        axs[0].plot(silhouette_score_sorted, '-')
        axs[0].set_ylabel(f'Silhouette score ({mode})', fontsize=7)
        axs[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        axs[0].spines["top"].set_visible(False)
        axs[0].spines["right"].set_visible(False)

    axs[1].plot(n_clusters_sorted, '-')
    axs[1].set_ylabel(f'Num. clusters ({mode})', fontsize=7)
    axs[1].set_yticks([0, 10, 20, 30])
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)

    axs[2].plot(avg_perf_train_list_sorted, '-')
    axs[2].set_ylabel('Avg. perf. train', fontsize=7)
    axs[2].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)

    axs[3].plot(avg_perf_test_list_sorted, '-')
    axs[3].set_ylabel('Avg. perf. test', fontsize=7)
    axs[3].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)

    im = axs[4].imshow(hp_visualize, aspect='auto', cmap='viridis')
    axs[4].set_yticks(range(len(hp_plots)))
    axs[4].set_yticklabels([HP_NAME[hp] for hp in hp_plots], fontsize=7)
    axs[4].set_xlabel('Networks')
    axs[4].tick_params(length=0)
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["right"].set_visible(False)
    axs[4].set_xticks([0, len(n_clusters_sorted) - 1])
    axs[4].set_xticklabels([1, len(n_clusters_sorted)])

    # === Add best model_dir text ===
    best_model_dir = '||'.join(successful_model_dirs_sorted[0].split('\\')[-3:])
    print('Best Model:      ', successful_model_dirs_sorted[0])

    # Save best model for later analysis
    if batchPlot == False:
        sortedModels_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"bestModels_{sort_variable}_{mode}.txt")
    else:
        sortedModels_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"bestModels_{sort_variable}_{mode}_{model_dir_batches[0]}.txt")

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(sortedModels_path), exist_ok=True)
    # Save best model path
    with open(sortedModels_path, "w") as f:
        # text_file.write(successful_model_dirs_sorted[0]) # fix Write all models in descending order into text file and make them callable like a list
        json.dump(successful_model_dirs_sorted, f, indent=2)

    fig.text(0.5, 0.95, f'highest {sort_variable} {mode} model: {best_model_dir}', ha='center', va='top', fontsize=8)

    # === Save figure ===
    if batchPlot == True:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"{model_dir_batches[0]}_{sort_variable}_{mode}_hp_plot.png")
    else:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"general_{sort_variable}_{mode}_hp_plot.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def plot_vertical_hp_legend(hp_ranges, hp_plots, HP_NAME, directory):
    cmap = mpl.cm.get_cmap('viridis')
    entries_per_hp = [len(hp_ranges[hp]) + 1 for hp in hp_plots]  # +1 for title
    total_lines = sum(entries_per_hp)

    line_height_in = 0.25
    fig_height_in = total_lines * line_height_in

    fig, ax = plt.subplots(figsize=(3.2, fig_height_in))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, total_lines)
    ax.axis("off")

    bar_width = 0.1
    bar_height = 0.8
    y = total_lines - 1

    for hp_name in hp_plots:
        label = HP_NAME.get(hp_name, hp_name)
        values = hp_ranges[hp_name]
        n = len(values)
        # print(label, values)

        ax.text(0.05, y + 0.3, f"{label}:", fontsize=9, fontweight='bold', va='top')
        y -= 1.0

        for j, val in enumerate(values):
            color = cmap(j / (n - 1) if n > 1 else 0.5)
            ax.add_patch(plt.Rectangle((0.05, y - bar_height / 2), bar_width, bar_height,
                                       facecolor=color, edgecolor='black', linewidth=0.3))
            ax.text(0.05 + bar_width + 0.05, y, val, fontsize=8, va='center')
            y -= 0.9

    # === Save figure ===
    save_path = os.path.join(directory, 'visuals', "legend_general_hp_plot.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    plt.show()

def _individual_hp_plot(hp_plot, sort_variable, mode, directory, batchPlot, model_dir_batches, n_clusters=None, silhouette_score=None, avg_perf_test_list=None, avg_perf_train_list=None, hp_list=None, modularity_list_sparse=None):
    """Plot histogram for number of clusters, separating by an attribute.

    Args:
        hp_plot: str, the attribute to separate histogram by
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    if hp_list is None: # attention: Maybe wrong fix here
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse = get_n_clusters(successful_model_dirs) # fix: variable still to deliver

    # Compare activation, ignore tanh that can not be trained with LeakyRNN
    # hp_plot = 'activation'
    # hp_plot = 'rnn_type'
    # hp_plot = 'w_rec_init'

    sort_variable_dict = OrderedDict()
    hp_ranges = _get_hp_ranges()
    for key in hp_ranges[hp_plot]:
        sort_variable_dict[key] = list()

    if sort_variable == 'performance':
        if mode == 'test':
            for hp, perf_test in zip(hp_list, avg_perf_test_list):
                sort_variable_dict[hp[hp_plot]].append(perf_test)
        elif mode == 'train':
            for hp, perf_train in zip(hp_list, avg_perf_train_list):
                sort_variable_dict[hp[hp_plot]].append(perf_train)

    elif sort_variable == 'clustering':
        for hp, n_cluster in zip(hp_list, n_clusters):
            sort_variable_dict[hp[hp_plot]].append(n_cluster)

    elif sort_variable == 'silhouette':
        for hp, silhouette in zip(hp_list, silhouette_score):
            sort_variable_dict[hp[hp_plot]].append(silhouette)

    elif sort_variable == 'modularity':
        for hp, modu in zip(hp_list, modularity_list_sparse):
            sort_variable_dict[hp[hp_plot]].append(modu)

    label_map = {'softplus': 'Softplus',
                 'relu': 'ReLU',
                 'tanh': 'Tanh',
                 'LeakyGRU': 'GRU',
                 'LeakyRNN': 'RNN',
                 'MultiLayer': 'MultiRNN',
                 'randortho': 'Rand. Ortho.',
                 'diag': 'Diagonal',
                 'randgauss': 'Rand. Gaussian',
                 'brainStructure': 'Brain Struc.',
                 'constant': 'Constant',
                 'exp_range': 'Exp. Range',
                 'triangular2': 'Triangular'
                 }

    # hp_ranges = OrderedDict()
    # hp_ranges['activation'] = ['softplus', 'relu', 'tanh']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU', 'MultiLayer']
    # hp_ranges['n_rnn'] = [128, 256, 512]
    # hp_ranges['w_rec_init'] = ['randortho', 'randgauss', 'diag', 'brainStructure']
    # hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_h'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['l2_weight'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['learning_rate'] = [0.0015, 0.001, 0.0005]
    # hp_ranges['learning_rate_mode'] = ['constant', 'exp_range', 'triangular2']

    # fig = plt.figure(figsize=(1.5, 1.2))
    # ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    f, axs = plt.subplots(len(sort_variable_dict), 1,
                          sharex=True, figsize=(1.2, 1.8))

    for i, (key, val) in enumerate(sort_variable_dict.items()):
        ax = axs[i]
        # hist, bin_edges = np.histogram(val, density=True, range=(0, 30),
        #                                bins=30)
        # plt.bar(bin_edges[:-1], hist, label=key)
        color_ind = i / (len(hp_ranges[hp_plot]) - 1.)
        color = mpl.cm.viridis(color_ind)
        if isinstance(key, float):
            label = '{:1.0e}'.format(key)
        else:
            label = label_map.get(key, str(key))

        if sort_variable == 'performance' or sort_variable == 'silhouette' or sort_variable == 'modularity':
            ax.hist(val, label=label, range=(0, 1),
                    density=True, bins=16, ec=color, facecolor=color,lw=1.5)
        elif sort_variable == 'clustering':
            ax.hist(val, label=label, range=(0, 30),
                    density=True, bins=16, ec=color, facecolor=color, lw=1.5)

        ax.spines["left"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_yticks([])

        if sort_variable == 'performance' or sort_variable == 'silhouette' or sort_variable == 'modularity':
            ax.set_xticks([0, 0.5, 1.0])
            ax.set_xlim([0, 1.0])
            ax.text(0.7, 0.7, label, fontsize=7, transform=ax.transAxes)
        elif sort_variable == 'clustering':
            ax.set_xticks([0, 15, 30])
            ax.set_xlim([0, 30])
            ax.text(0.7, 0.7, label, fontsize=7, transform=ax.transAxes)

        if i == 0:
            ax.set_title(HP_NAME[hp_plot], fontsize=7)
    # ax.legend(loc=3, bbox_to_anchor=(1, 0), title=HP_NAME[hp_plot], frameon=False)
    if sort_variable == 'performance':
        ax.set_xlabel('Performance', fontsize=7)
    elif sort_variable == 'clustering':
        ax.set_xlabel('Number of clusters', fontsize=7)
    elif sort_variable == 'silhouette':
        ax.set_ylabel('Silhouette score', fontsize=7)
    elif sort_variable == 'modularity':
        ax.set_ylabel('Modularity score', fontsize=7)

    # plt.tight_layout()
    # plt.show()
    # figname = os.path.join(FIGPATH, 'NumClustersHist' + hp_plot + '.pdf')
    # plt.savefig(figname, transparent=True)

        # === Save figure ===
    if batchPlot == True:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', 'batchPlots', model_dir_batches[0], f"{model_dir_batches[0]}_{hp_plot}_{mode}_plot.png")
    else:
        save_path = os.path.join(directory, 'visuals', f'{sort_variable}_{mode}', f"general_{hp_plot}_{mode}_plot.png")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return sort_variable_dict

def individual_hp_plot(n_clusters, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse, directory, hp_list, sort_variable, mode, batchPlot, model_dir_batches):
    """Plot histogram of number of clusters.

    Args:
        n_clusters: list of cluster numbers
        hp_list: list of hp dictionary
    """
    hp_plots = ['activation', 'n_rnn', 'w_rec_init', 'l1_h', 'l1_weight', 'l2_h', 'l2_weight', 'learning_rate', 'learning_rate_mode']
    # hp_plots = ['activation', 'rnn_type', 'n_rnn', 'w_rec_init', 'l1_h', 'l1_weight', 'l2_h', 'l2_weight', 'learning_rate', 'learning_rate_mode', 'errorBalancingValue']

    for hp_plot in hp_plots:
        n_cluster_dict = _individual_hp_plot(hp_plot, sort_variable, mode, directory, batchPlot, model_dir_batches, n_clusters, silhouette_score, avg_perf_test_list, avg_perf_train_list, hp_list, modularity_list_sparse)

# fix: Add network size here please
HP_NAME = {'activation': 'Activation fun.',
           # 'rnn_type': 'Network type',
           'w_rec_init': 'Initialization',
           'n_rnn': 'Num. hidden units',
           'l1_h': 'L1 rate',
           'l1_weight': 'L1 weight',
           'l2_h': 'L2 rate',
           'l2_weight': 'L2 weight',
           'target_perf': 'Target perf.',
           'learning_rate': 'Learning rate',
           'learning_rate_mode': 'Learning rate mode'}
           # 'errorBalancingValue': 'Error balancing value'}

if __name__ == '__main__':
    folderList = [r'__paper1_placeholders\brainInitialization_brainMasking_test_4task']
    for folder in folderList:
        final_model_dirs = []

        participant = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05'][0]
        dataType = ['highDim', 'highDim_3stimTC', 'highDim_correctOnly'][2]

        mode = ['train', 'test'][1]
        sort_variable = ['clustering', 'performance', 'silhouette'][1]
        batchPlot = [True, False][1]
        lastMonth = '6'

        directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\{folder}\{dataType}\{participant}'

        if batchPlot == False:
            model_dir_batches = os.listdir(directory)
        else:
            model_dir_batches = ['1'] # info: For creating a hp overview for one batch (e.g. in robustnessTest)

        # Create list of models to integrate in one hp overview plot
        model_dir_batches = [batch for batch in model_dir_batches if batch != 'visuals']
        for model_dir_batch in model_dir_batches:
            model_dirs_ = os.listdir(os.path.join(directory, model_dir_batch))
            model_dirs = [model_dir for model_dir in model_dirs_ if model_dir != 'overviews' and not model_dir.endswith('.txt')]
            for model_dir in model_dirs:
                model_dir_lastMonth_ = os.listdir(os.path.join(directory, model_dir_batch, model_dir))
                model_dir_lastMonth = [model_dir for model_dir in model_dir_lastMonth_ if lastMonth in model_dir]
                # Concatenate all models in one list
                try:
                    if 'model' in model_dir_lastMonth[0]: # Be sure to add anything else but models
                        final_model_dirs.append(os.path.join(directory, model_dir_batch, model_dir, model_dir_lastMonth[0]))

                except Exception as e:
                    # if something goes wrong (e.g. index error), skip this model_dir
                    print(f"Skipping {model_dir} due to error: {e}")
                    continue

        # First compute n_clusters for each model then collect them in lists
        successful_model_dirs = compute_n_cluster(final_model_dirs, mode)
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse = get_n_clusters(successful_model_dirs)

        # Create hp_plots sorted by performance or clustering
        general_hp_plot(n_clusters, silhouette_score, hp_list, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse, directory, sort_variable, mode, batchPlot, model_dir_batches)
        # Create legend
        hp_ranges = _get_hp_ranges()
        hp_plots = list(hp_ranges.keys())
        plot_vertical_hp_legend(hp_ranges, hp_plots, HP_NAME, directory)

        # Create histogramms for each hyperparameter seperatly w.r.t. performance or clustering
        individual_hp_plot(n_clusters, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse, directory, hp_list, sort_variable, mode, batchPlot, model_dir_batches)


# info: DEBUG
# import os
# folderList = ['_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_16', '_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_32', '_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_64',
#                   '_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_128', '_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_256', '_gridSearch_domainTask-WM_beRNN_03_highDim_correctOnly_512']
#
# participant = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05'][2]
# dataType = ['highDim', 'highDim_3stimTC', 'highDim_correctOnly'][2]
#
# for folder in folderList:
#     directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\{folder}\{dataType}\{participant}'
#     counter = 0
#     for modelBatch in os.listdir(directory):
#         for model in os.listdir(os.path.join(directory, modelBatch)):
#             if model == 'times.txt':
#                 continue
#             if len(os.listdir(os.path.join(directory, modelBatch, model, 'model_month_6'))) < 3:
#                 counter += 1
#     print(folder)
#     print(counter)
#     print('******************************')



# # Robustness tests for topological marker
# def apply_density_threshold(matrix, density=0.1):
#     """
#     Applies proportional (density) thresholding to keep the top X%
#     of edges globally based on absolute strength (either strong positive or negative).
#
#     Args:
#         matrix (np.ndarray): Input symmetric correlation matrix.
#         density (float): The proportion of edges to keep (e.g., 0.1 for 10%).
#
#     Returns:
#         np.ndarray: The thresholded matrix with strong correlations retained.
#     """
#     # info: modularity function aligned with training pipeline - 17.11.25
#     # Use a copy to avoid modifying the original input matrix
#     temp_matrix = matrix.copy()
#     n = temp_matrix.shape[0]
#
#     # Ensure diagonal is 0 for edge calculation
#     np.fill_diagonal(temp_matrix, 0)
#
#     upper_tri_indices = np.triu_indices(n, k=1)
#     triu_vals_signed = temp_matrix[upper_tri_indices]
#     abs_triu_vals = np.abs(triu_vals_signed)
#
#     cutoff = np.quantile(abs_triu_vals, 1 - density)
#     thresholded = np.where(np.abs(matrix) >= cutoff, matrix, 0)
#
#     # Ensure diagonal remains 0
#     np.fill_diagonal(thresholded, 0)
#
#     return thresholded
#
# import os
# import pickle
# import networkx as nx
# import numpy as np
#
# avg_clustering_list = []
# avg_betweenness_list = []
# avg_closeness_list = []
#
# participant = 'beRNN_01'
# hp_set = '2'
# models = os.listdir(fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_robustnessTest_multiTask_{participant}_highDimCorrects_256_hp_{hp_set}\highDim_correctOnly\{participant}\{hp_set}')
#
# correlationMatrixList = []
#
# for model in models:
#     if model == 'times.txt':
#         continue
#     else:
#         pkl_beRNN = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_robustnessTest_multiTask_{participant}_highDimCorrects_256_hp_{hp_set}\highDim_correctOnly\{participant}\{hp_set}\{model}\model_month_6\corr_test_lay1_rule_all.pkl'
#         # print(pkl_beRNN)
#         with open(pkl_beRNN, 'rb') as f:
#             correlationMatrix_taskwise = pickle.load(f)
#             np.nan_to_num(correlationMatrix_taskwise['h_corr_all'], copy=False, nan=0)
#             correlationMatrix = np.mean(correlationMatrix_taskwise['h_corr_all'], axis=2)
#
#             correlationMatrixList.append(correlationMatrix)
#
#             # # Define a threshold (you can experiment with this value)
#             # threshold = .1
#             # # averaged_correlation_matrix_thresholded = apply_absolute_threshold(matrix_brain, threshold)
#             # averaged_correlation_matrix_thresholded = apply_density_threshold(correlationMatrix, threshold)
#             # np.fill_diagonal(averaged_correlation_matrix_thresholded, 0)  # prevent self-loops
#             #
#             # # Function to apply a threshold to the matrix
#             # G_beRNN = nx.from_numpy_array(averaged_correlation_matrix_thresholded)
#             #
#             # clustering = nx.clustering(G_beRNN)
#             # avg_clustering = np.mean(list(clustering.values()))
#             # avg_clustering_list.append(avg_clustering)
#             #
#             # betweenness = nx.betweenness_centrality(G_beRNN)
#             # avg_betweenness = np.mean(list(betweenness.values()))
#             # avg_betweenness_list.append(avg_betweenness)
#             #
#             # closeness = nx.closeness_centrality(G_beRNN)  # Closeness centrality measures the average distance from a node to all other nodes in the network.
#             # avg_closeness = np.mean(list(closeness.values()))
#             # avg_closeness_list.append(avg_closeness)
#
# std_clustering = np.std(avg_clustering_list)
# print(std_clustering)
# std_betweenness = np.std(avg_betweenness_list)
# print(std_betweenness)
# std_closeness = np.std(avg_closeness_list)
# print(std_closeness)
#
# print('************************************')
#
# avg_clustering = np.mean(avg_clustering_list)
# print(avg_clustering)
# avg_betweenness = np.mean(avg_betweenness_list)
# print(avg_betweenness)
# avg_closeness = np.mean(avg_closeness_list)
# print(avg_closeness)




###
# import numpy as np
# matrix = np.load(r'W:\group_csp\analyses\oliver.frank\share\functional_matrices\sub-6IECX_ses-01-task-faces-atlas-4S256Parcels.npy')