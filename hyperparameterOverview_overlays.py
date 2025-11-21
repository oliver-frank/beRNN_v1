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
from matplotlib.lines import Line2D

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
    # hp_ranges['activation'] = ['softplus', 'relu', 'tanh']
    # hp_ranges['rnn_type'] = ['LeakyRNN', 'LeakyGRU', 'MultiLayer']
    # hp_ranges['n_rnn'] = [8, 16, 24, 32, 48, 64, 96, 128, 256, 512]
    # hp_ranges['w_rec_init'] = ['randortho', 'randgauss', 'diag', 'brainStructure']
    hp_ranges['l1_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l1_weight'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_h'] = [0, 1e-5, 1e-4, 1e-3]
    hp_ranges['l2_weight'] = [0, 1e-5, 1e-4, 1e-3]
    # hp_ranges['learning_rate'] = [0.002, 0.0015, 0.001, 0.0005, 0.0001, 0.00005]
    # hp_ranges['learning_rate_mode'] = ['constant', 'exp_range', 'triangular2']
    # hp_ranges['errorBalancingValue'] = [1., 5.]
    return hp_ranges

def general_hp_plot_overlay_multiple(meta_n_clusters_list,
                                     meta_silhouette_score_list,
                                     meta_hp_list,
                                     meta_perf_train_list,
                                     meta_perf_test_list,
                                     meta_modularity_list,
                                     models_labels,
                                     directory,
                                     sort_variable='performance',
                                     mode='test',
                                     alpha=0.6,
                                     cmap_name='viridis'):
    """
    Overlay multiple grid-searches onto one plot.
    Each element in meta_* lists corresponds to one grid-search (list of values per model).
    - models_labels: list of strings used in legend (same order as meta lists)
    - alpha: transparency for overlays
    """

    # Basic checks
    n_searches = len(meta_hp_list)
    assert len(models_labels) == n_searches, "models_labels must match number of meta entries"

    hp_ranges = _get_hp_ranges()
    hp_plots = list(hp_ranges.keys())

    # choose distinct colors
    cmap_colors = mpl.cm.get_cmap('Blues')
    colors = [cmap_colors(0.4 + 0.6 * i / max(1, n_searches - 1)) for i in range(n_searches)]

    # prepare figure
    fig, axs = plt.subplots(5, 1, figsize=(7, 6.5), sharex=True,
                            gridspec_kw={'height_ratios': [1, 1, 1, 1, 1.2]})
    plt.subplots_adjust(hspace=0.5)

    hp_visualize_list = []
    total_models = 0  # for x-axis labeling later

    for s in range(n_searches):
        n_clusters = list(meta_n_clusters_list[s])
        silhouette_score = list(meta_silhouette_score_list[s])
        hp_list = list(meta_hp_list[s])
        avg_perf_train_list = list(meta_perf_train_list[s])
        avg_perf_test_list = list(meta_perf_test_list[s])
        modularity_list = list(meta_modularity_list[s]) if meta_modularity_list[s] is not None else [0] * len(n_clusters)

        # et negative modularity scores to 0
        modularity_list = [max(0, m) for m in modularity_list]

        # Sorting
        if sort_variable == 'performance' and mode == 'test':
            ind_sort = np.argsort(avg_perf_test_list)[::-1]
        elif sort_variable == 'performance' and mode == 'train':
            ind_sort = np.argsort(avg_perf_train_list)[::-1]
        elif sort_variable == 'clustering':
            ind_sort = np.argsort(n_clusters)[::-1]
        elif sort_variable == 'silhouette':
            ind_sort = np.argsort(silhouette_score)[::-1]
        elif sort_variable == 'modularity':
            ind_sort = np.argsort(modularity_list)[::-1]
        else:
            ind_sort = np.arange(len(n_clusters))

        n_clusters_sorted = np.array(n_clusters)[ind_sort]
        silhouette_sorted = np.array(silhouette_score)[ind_sort]
        perf_train_sorted = np.array(avg_perf_train_list)[ind_sort]
        perf_test_sorted = np.array(avg_perf_test_list)[ind_sort]
        modularity_sorted = np.array(modularity_list)[ind_sort]
        hp_list_sorted = [hp_list[i] for i in ind_sort]
        x = np.arange(len(n_clusters_sorted))

        total_models = max(total_models, len(x))

        # Plot main metrics
        if hp_list_sorted and hp_list_sorted[0].get('rnn_type') != 'MultiLayer':
            axs[0].plot(x, modularity_sorted, '-', alpha=alpha, color=colors[s], label=models_labels[s])
        else:
            axs[0].plot(x, silhouette_sorted, '-', alpha=alpha, color=colors[s], label=models_labels[s])

        axs[1].plot(x, n_clusters_sorted, '-', alpha=alpha, color=colors[s])
        axs[2].plot(x, perf_train_sorted, '-', alpha=alpha, color=colors[s])
        axs[3].plot(x, perf_test_sorted, '-', alpha=alpha, color=colors[s])

        # Build discrete hp_visualize array (no normalization)
        hp_visualize = np.zeros((len(hp_plots), len(hp_list_sorted)))
        for i_hp, hp_name in enumerate(hp_plots):
            unique_vals = hp_ranges[hp_name]
            for j_model, hp in enumerate(hp_list_sorted):
                val = hp.get(hp_name, None)
                if val in unique_vals:
                    hp_visualize[i_hp, j_model] = unique_vals.index(val)
                else:
                    try:
                        uniq_nums = [uv for uv in unique_vals if isinstance(uv, (int, float))]
                        if isinstance(val, (int, float)) and uniq_nums:
                            idx = np.argmin(np.abs(np.array(uniq_nums) - val))
                            hp_visualize[i_hp, j_model] = idx
                        else:
                            hp_visualize[i_hp, j_model] = 0
                    except Exception:
                        hp_visualize[i_hp, j_model] = 0
        hp_visualize_list.append(hp_visualize)

    # Format upper plots
    if meta_hp_list and meta_hp_list[0] and meta_hp_list[0][0].get('rnn_type') != 'MultiLayer':
        axs[0].set_ylabel(f'Mod. score ({mode})', fontsize=8)
    else:
        axs[0].set_ylabel(f'Silhouette score ({mode})', fontsize=8)
    axs[1].set_ylabel(f'Num. clusters ({mode})', fontsize=8)
    axs[2].set_ylabel('Avg. perf. train', fontsize=8)
    axs[3].set_ylabel('Avg. perf. test', fontsize=8)

    axs[0].set_yticks([0.0, 0.5, 1.0])
    axs[0].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[1].set_yticks([0, 15, 30])
    axs[1].set_yticklabels(["0", "15", "30"])
    axs[2].set_yticks([0.0, 0.5, 1.0])
    axs[2].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[3].set_yticks([0.0, 0.5, 1.0])
    axs[3].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)

    axs[0].legend(models_labels, fontsize=8, loc='best', frameon=False)

    # HP overlay as dots
    axs[4].set_yticks(range(len(hp_plots)))
    axs[4].set_yticklabels([HP_NAME.get(hp, hp) for hp in hp_plots], fontsize=7)
    axs[4].set_xlabel('Model rank')
    axs[4].tick_params(length=0)
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    cmap = plt.get_cmap(cmap_name)

    for s in range(n_searches):
        hp_visualize = hp_visualize_list[s]
        x = np.arange(hp_visualize.shape[1])
        for i_hp in range(len(hp_plots)):
            y = np.full_like(x, i_hp)
            cvals = hp_visualize[i_hp, :].astype(int)
            axs[4].scatter(
                x, y + np.random.uniform(-0.15, 0.15, size=len(x)),
                c=[cmap(c / max(1, len(hp_ranges[hp_plots[i_hp]]) - 1)) for c in cvals],
                s=12, alpha=alpha, edgecolor='none'
            )

    axs[4].set_ylim(-0.5, len(hp_plots) - 0.5)
    axs[4].invert_yaxis()

    # === Legends ===
    # (1) Discrete HP legend for L1 rate
    legend_elements = []
    l1_vals = hp_ranges['l1_h']
    for i, val in enumerate(l1_vals):
        color = cmap(i / max(1, len(l1_vals) - 1))
        legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                      markerfacecolor=color, markersize=6,
                                      label=f"{val:.0e}"))

    reg_legend = axs[4].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                               loc='upper left', fontsize=6, frameon=False, title="Reg. values")

    # (2) Network size legend (matching top-plot colors, no box)
    network_sizes = ['16', '32', '64', '128', '256', '512']

    # Use same color cycle as used in the main plots
    line_colors = colors[:len(network_sizes)]

    # Create line legend handles (matching style and colors)
    net_legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=size)
        for color, size in zip(line_colors, network_sizes)
    ]

    # Add legend in top-right corner, same font as reg. legend, no box
    net_legend = axs[3].legend(
        handles=net_legend_handles,
        bbox_to_anchor=(1.05, 1),
        title='Network size',
        fontsize=6,
        loc='upper left',
        frameon=False
    )

    # Keep reg. legend visible
    axs[4].add_artist(reg_legend)

    # Show total number of models on x-axis
    axs[4].set_xlim(0, total_models)
    axs[4].set_xticks([0, total_models])
    axs[4].set_xticklabels(['0', f'{total_models}'])

    # === Save ===
    save_path = os.path.join(directory, 'visuals',
                             f'overlay_multi_{"-".join(models_labels[0].split("_")[1:-1])}_{sort_variable}_{mode}.png')
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

    foldersToOverlay = ['_gridSearch_multiTask_beRNN_03_highDimCorrects_16', '_gridSearch_multiTask_beRNN_03_highDimCorrects_32', '_gridSearch_multiTask_beRNN_03_highDimCorrects_64',\
                        '_gridSearch_multiTask_beRNN_03_highDimCorrects_128', '_gridSearch_multiTask_beRNN_03_highDimCorrects_256', '_gridSearch_multiTask_beRNN_03_highDim_correctOnly_512']

    directory_metaOverlayVisual = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__metaOverlayVisual'
    os.makedirs(directory_metaOverlayVisual, exist_ok=True)

    meta_silhouette_score_list = list()
    meta_modularity_list = list()
    meta_n_clusters_list = list()
    meta_perf_train_list = list()
    meta_perf_test_list = list()
    meta_hp_list = list()

    for folder in foldersToOverlay:
        final_model_dirs = []

        participant = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05'][2]
        dataType = ['highDim', 'highDim_3stimTC', 'highDim_correctOnly'][2]

        mode = ['train', 'test'][1]
        sort_variable = ['clustering', 'performance', 'silhouette'][1]
        batchPlot = [True, False][1]
        lastMonth = '6'

        directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\{folder}\{dataType}\{participant}'

        if batchPlot == False:
            model_dir_batches = os.listdir(directory)
        else:
            model_dir_batches = ['4']  # info: For creating a hp overview for one batch (e.g. in robustnessTest)

        # Create list of models to integrate in one hp overview plot
        model_dir_batches = [batch for batch in model_dir_batches if batch != 'visuals']
        for model_dir_batch in model_dir_batches:
            model_dirs_ = os.listdir(os.path.join(directory, model_dir_batch))
            model_dirs = [model_dir for model_dir in model_dirs_ if
                          model_dir != 'overviews' and not model_dir.endswith('.txt')]
            for model_dir in model_dirs:
                model_dir_lastMonth_ = os.listdir(os.path.join(directory, model_dir_batch, model_dir))
                model_dir_lastMonth = [model_dir for model_dir in model_dir_lastMonth_ if lastMonth in model_dir]
                # Concatenate all models in one list
                try:
                    if 'model' in model_dir_lastMonth[0]:  # Be sure to add anything else but models
                        final_model_dirs.append(os.path.join(directory, model_dir_batch, model_dir, model_dir_lastMonth[0]))

                except Exception as e:
                    # if something goes wrong (e.g. index error), skip this model_dir
                    print(f"Skipping {model_dir} due to error: {e}")
                    continue

        # First compute n_clusters for each model then collect them in lists
        successful_model_dirs = compute_n_cluster(final_model_dirs, mode)
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, modularity_list_sparse = get_n_clusters(
            successful_model_dirs)

        # info: After everything was done as in hyperparameterOverview.py, catch the important lists and save them to meta lists
        meta_silhouette_score_list.append(silhouette_score)
        meta_modularity_list.append(modularity_list_sparse)
        meta_n_clusters_list.append(n_clusters)
        meta_perf_train_list.append(avg_perf_train_list)
        meta_perf_test_list.append(avg_perf_test_list)
        meta_hp_list.append(hp_list)

    import json
    with open(os.path.join(directory_metaOverlayVisual, 'meta_silhouette_score_list.json'), 'w') as f:
        json.dump(meta_silhouette_score_list, f)
    with open(os.path.join(directory_metaOverlayVisual, 'meta_modularity_list.json'), 'w') as f:
        json.dump(meta_modularity_list, f)
    with open(os.path.join(directory_metaOverlayVisual, 'meta_n_clusters_list.json'), 'w') as f:
        json.dump(meta_n_clusters_list, f)
    with open(os.path.join(directory_metaOverlayVisual, 'meta_perf_train_list.json'), 'w') as f:
        json.dump(meta_perf_train_list, f)
    with open(os.path.join(directory_metaOverlayVisual, 'meta_perf_test_list.json'), 'w') as f:
        json.dump(meta_perf_test_list, f)
    with open(os.path.join(directory_metaOverlayVisual, 'meta_hp_list.json'), 'w') as f:
        json.dump(meta_hp_list, f)

    # Visualize big overlay
    # general_hp_plot_overlay(n_clusters, silhouette_score, hp_list, avg_perf_train_list, avg_perf_test_list,
    #                 modularity_list_sparse, directory, sort_variable, mode, batchPlot, model_dir_batches)

    general_hp_plot_overlay_multiple(meta_n_clusters_list,
                                     meta_silhouette_score_list,
                                     meta_hp_list,
                                     meta_perf_train_list,
                                     meta_perf_test_list,
                                     meta_modularity_list,
                                     foldersToOverlay,
                                     directory_metaOverlayVisual,
                                     sort_variable='performance',
                                     mode='test',
                                     alpha=0.6,
                                     cmap_name='viridis')



