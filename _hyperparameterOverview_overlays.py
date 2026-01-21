########################################################################################################################
# head: hp overview ####################################################################################################
########################################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from collections import OrderedDict
from scipy.stats import pearsonr
import networkx as nx

from _hyperparameterOverview import compute_n_cluster, get_n_clusters #, plot_vertical_hp_legend


########################################################################################################################
# head: Create histogramms to visualize and investigate interrelations of hyperparameter, modularity and performance ###
########################################################################################################################
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

# Visualizes overlay of several grid searches for one participant
def general_hp_plot_overlay_multiple(meta_n_clusters_list,
                                     meta_silhouette_score_list,
                                     meta_hp_list,
                                     meta_perf_train_list,
                                     meta_perf_test_list,
                                     meta_clustering_list,
                                     meta_modularity_list,
                                     meta_participation_coefficient_list,
                                     folder_labels,
                                     directory,
                                     density,
                                     network_sizes,
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
    assert len(folder_labels) == n_searches, "folder_labels must match number of meta entries"

    hp_ranges = _get_hp_ranges()
    hp_plots = list(hp_ranges.keys())

    # choose distinct colors
    cmap_colors = mpl.cm.get_cmap('Blues')
    colors = [cmap_colors(0.4 + 0.6 * i / max(1, n_searches - 1)) for i in range(n_searches)]

    # prepare figure
    fig, axs = plt.subplots(6, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1, 1.5]})
    plt.subplots_adjust(hspace=0.5)

    hp_visualize_list = []
    total_models = 0  # for x-axis labeling later

    for s in range(n_searches):
        n_clusters = list(meta_n_clusters_list[s])
        silhouette_score = list(meta_silhouette_score_list[s])
        hp_list = list(meta_hp_list[s])
        avg_perf_train_list = list(meta_perf_train_list[s])
        avg_perf_test_list = list(meta_perf_test_list[s])
        clustering_list = list(meta_clustering_list[s]) if meta_clustering_list[s] is not None else [0] * len(n_clusters)
        modularity_list = list(meta_modularity_list[s]) if meta_modularity_list[s] is not None else [0] * len(n_clusters)
        participation_list = list(meta_participation_coefficient_list[s]) if meta_participation_coefficient_list[s] is not None else [0] * len(n_clusters)

        # get negative modularity scores to 0 as they occur in ~0 cases
        modularity_list = [max(0, m) for m in modularity_list]

        # Sorting
        if sort_variable == 'performance' and mode == 'test':
            ind_sort = np.argsort(avg_perf_test_list)[::-1]
        elif sort_variable == 'performance' and mode == 'train':
            ind_sort = np.argsort(avg_perf_train_list)[::-1]
        # elif sort_variable == 'clustering':
        #     ind_sort = np.argsort(n_clusters)[::-1]
        # elif sort_variable == 'silhouette':
        #     ind_sort = np.argsort(silhouette_score)[::-1]
        # elif sort_variable == 'modularity':
        #     ind_sort = np.argsort(modularity_list)[::-1]
        # else:
        #     ind_sort = np.arange(len(n_clusters))

        n_clusters_sorted = np.array(n_clusters)[ind_sort]
        silhouette_sorted = np.array(silhouette_score)[ind_sort]
        perf_train_sorted = np.array(avg_perf_train_list)[ind_sort]
        perf_test_sorted = np.array(avg_perf_test_list)[ind_sort]
        clustering_sorted = np.array(clustering_list)[ind_sort]
        modularity_sorted = np.array(modularity_list)[ind_sort]
        participation_sorted = np.array(participation_list)[ind_sort]
        hp_list_sorted = [hp_list[i] for i in ind_sort]
        x = np.arange(len(n_clusters_sorted))

        total_models = max(total_models, len(x))

        # Plot main metrics
        if hp_list_sorted and hp_list_sorted[0].get('rnn_type') != 'MultiLayer':
            axs[0].plot(x, clustering_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s])
            axs[1].plot(x, modularity_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s])
            axs[2].plot(x, participation_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s])
        else:
            axs[0].plot(x, silhouette_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s])

        # axs[3].plot(x, n_clusters_sorted, '-', alpha=alpha, color=colors[s])
        axs[3].plot(x, perf_train_sorted, '-', alpha=alpha, color=colors[s])
        axs[4].plot(x, perf_test_sorted, '-', alpha=alpha, color=colors[s])

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
        axs[0].set_ylabel(f'Clustering', fontsize=8)
        axs[1].set_ylabel(f'Modularity', fontsize=8)
        axs[2].set_ylabel(f'Participation', fontsize=8)
    else:
        axs[0].set_ylabel(f'Silhouette score', fontsize=8)

    axs[3].set_ylabel('Avg. perf. train', fontsize=8)
    axs[4].set_ylabel('Avg. perf. test', fontsize=8)

    # axs[1].set_yticks([0, 15, 30])
    # axs[1].set_yticklabels(["0", "15", "30"])
    axs[0].set_yticks([0.0, 0.5, 1.0])
    axs[0].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[1].set_yticks([0.0, 0.5, 1.0])
    axs[1].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[2].set_yticks([0.0, 0.5, 1.0])
    axs[2].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[3].set_yticks([0.0, 0.5, 1.0])
    axs[3].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[4].set_yticks([0.0, 0.5, 1.0])
    axs[4].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    axs[0].legend(folder_labels, fontsize=8, loc='best', frameon=False)

    # HP overlay as dots
    axs[5].set_yticks(range(len(hp_plots)))
    axs[5].set_yticklabels([HP_NAME.get(hp, hp) for hp in hp_plots], fontsize=7)
    axs[5].set_xlabel('Model rank')
    axs[5].tick_params(length=0)
    axs[5].spines["top"].set_visible(False)
    axs[5].spines["right"].set_visible(False)

    cmap = plt.get_cmap(cmap_name)

    for s in range(n_searches):
        hp_visualize = hp_visualize_list[s]
        x = np.arange(hp_visualize.shape[1])
        for i_hp in range(len(hp_plots)):
            y = np.full_like(x, i_hp)
            cvals = hp_visualize[i_hp, :].astype(int)
            axs[5].scatter(
                x, y + np.random.uniform(-0.15, 0.15, size=len(x)),
                c=[cmap(c / max(1, len(hp_ranges[hp_plots[i_hp]]) - 1)) for c in cvals],
                s=12, alpha=alpha, edgecolor='none'
            )

    axs[5].set_ylim(-0.5, len(hp_plots) - 0.5)
    axs[5].invert_yaxis()

    # === Legends ===
    # (1) Discrete HP legend for L1 rate
    legend_elements = []
    l1_vals = hp_ranges['l1_h']
    for i, val in enumerate(l1_vals):
        color = cmap(i / max(1, len(l1_vals) - 1))
        legend_elements.append(Line2D([0], [0], marker='o', color='none',
                                      markerfacecolor=color, markersize=6,
                                      label=f"{val:.0e}"))

    reg_legend = axs[5].legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
                               loc='upper left', fontsize=6, frameon=False, title="Reg. values")

    # Use same color cycle as used in the main plots
    line_colors = colors[:len(network_sizes)]

    # Create line legend handles (matching style and colors)
    net_legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=size)
        for color, size in zip(line_colors, network_sizes)
    ]

    # Add legend in top-right corner, same font as reg. legend, no box
    net_legend = axs[4].legend(
        handles=net_legend_handles,
        bbox_to_anchor=(1.05, 1),
        title='Network size',
        fontsize=6,
        loc='upper left',
        frameon=False
    )

    # Keep reg. legend visible
    axs[5].add_artist(reg_legend)

    # Show total number of models on x-axis
    axs[5].set_xlim(0, total_models)
    axs[5].set_xticks([0, total_models])
    axs[5].set_xticklabels(['0', f'{total_models}'])

    # === Save ===
    save_path = os.path.join(directory, 'visuals_overlay',
                             f'overlay_multi_density_{density}_{"-".join(folder_labels[0].split("_")[1:-1])}_{sort_variable}_{mode}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()
    # plt.close()

# Visualizes overlay of one robustness test for several participants
def general_hp_plot_overlay_multiple_robustnessTests(meta_n_clusters_list,
                                     meta_silhouette_score_list,
                                     meta_hp_list,
                                     meta_perf_train_list,
                                     meta_perf_test_list,
                                     meta_clustering_list,
                                     meta_modularity_list,
                                     meta_participation_coefficient_list,
                                     folder_labels,
                                     directory,
                                     density,
                                     participants,
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
    assert len(folder_labels) == n_searches, "models_labels must match number of meta entries"

    # hp_ranges = _get_hp_ranges()
    # hp_plots = list(hp_ranges.keys())

    # choose distinct colors
    cmap_colors = mpl.cm.get_cmap('Greys')
    colors = [cmap_colors(0.4 + 0.6 * i / max(1, n_searches - 1)) for i in range(n_searches)]

    # prepare figure
    fig, axs = plt.subplots(5, 1, figsize=(2, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 1, 1]})
    plt.subplots_adjust(hspace=0.5)

    # hp_visualize_list = []
    total_models = 0  # for x-axis labeling later

    for s in range(n_searches):
        n_clusters = list(meta_n_clusters_list[s])
        silhouette_score = list(meta_silhouette_score_list[s])
        hp_list = list(meta_hp_list[s])
        avg_perf_train_list = list(meta_perf_train_list[s])
        avg_perf_test_list = list(meta_perf_test_list[s])
        clustering_list = list(meta_clustering_list[s]) if meta_clustering_list[s] is not None else [0] * len(n_clusters)
        modularity_list = list(meta_modularity_list[s]) if meta_modularity_list[s] is not None else [0] * len(n_clusters)
        participation_list = list(meta_participation_coefficient_list[s]) if meta_participation_coefficient_list[s] is not None else [0] * len(n_clusters)

        # get negative modularity scores to 0 as they occur in ~0 cases
        modularity_list = [max(0, m) for m in modularity_list]

        # Sorting
        if sort_variable == 'performance' and mode == 'test':
            ind_sort = np.argsort(avg_perf_test_list)[::-1]
        elif sort_variable == 'performance' and mode == 'train':
            ind_sort = np.argsort(avg_perf_train_list)[::-1]
        # elif sort_variable == 'clustering':
        #     ind_sort = np.argsort(n_clusters)[::-1]
        # elif sort_variable == 'silhouette':
        #     ind_sort = np.argsort(silhouette_score)[::-1]
        # elif sort_variable == 'modularity':
        #     ind_sort = np.argsort(modularity_list)[::-1]
        # else:
        #     ind_sort = np.arange(len(n_clusters))

        n_clusters_sorted = np.array(n_clusters)[ind_sort]
        silhouette_sorted = np.array(silhouette_score)[ind_sort]
        perf_train_sorted = np.array(avg_perf_train_list)[ind_sort]
        perf_test_sorted = np.array(avg_perf_test_list)[ind_sort]
        clustering_sorted = np.array(clustering_list)[ind_sort]
        modularity_sorted = np.array(modularity_list)[ind_sort]
        participation_sorted = np.array(participation_list)[ind_sort]
        hp_list_sorted = [hp_list[i] for i in ind_sort]
        x = np.arange(len(n_clusters_sorted))

        total_models = max(total_models, len(x))

        # Plot main metrics
        if hp_list_sorted and hp_list_sorted[0].get('rnn_type') != 'MultiLayer':
            axs[0].plot(x, clustering_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s])
        else:
            axs[0].plot(x, silhouette_sorted, '-', alpha=alpha, color=colors[s], label=folder_labels[s]) # legacy

        axs[1].plot(x, modularity_sorted, '-', alpha=alpha, color=colors[s])
        axs[2].plot(x, participation_sorted, '-', alpha=alpha, color=colors[s])
        axs[3].plot(x, perf_train_sorted, '-', alpha=alpha, color=colors[s])
        axs[4].plot(x, perf_test_sorted, '-', alpha=alpha, color=colors[s])

        # # Build discrete hp_visualize array for later visualization (no normalization)
        # hp_visualize = np.zeros((len(hp_plots), len(hp_list_sorted)))
        # for i_hp, hp_name in enumerate(hp_plots):
        #     unique_vals = hp_ranges[hp_name]
        #     for j_model, hp in enumerate(hp_list_sorted):
        #         val = hp.get(hp_name, None)
        #         if val in unique_vals:
        #             hp_visualize[i_hp, j_model] = unique_vals.index(val)
        #         else:
        #             try:
        #                 uniq_nums = [uv for uv in unique_vals if isinstance(uv, (int, float))]
        #                 if isinstance(val, (int, float)) and uniq_nums:
        #                     idx = np.argmin(np.abs(np.array(uniq_nums) - val))
        #                     hp_visualize[i_hp, j_model] = idx
        #                 else:
        #                     hp_visualize[i_hp, j_model] = 0
        #             except Exception:
        #                 hp_visualize[i_hp, j_model] = 0
        # hp_visualize_list.append(hp_visualize)

    # Format upper plots
    if meta_hp_list and meta_hp_list[0] and meta_hp_list[0][0].get('rnn_type') != 'MultiLayer':
        axs[0].set_ylabel(f'Clustering', fontsize=8)
        axs[1].set_ylabel(f'Modularity', fontsize=8)
        axs[2].set_ylabel(f'Participation', fontsize=8)
    else:
        axs[0].set_ylabel(f'Silhouette score', fontsize=8)

    axs[3].set_ylabel('Avg. perf. train', fontsize=8)
    axs[4].set_ylabel('Avg. perf. test', fontsize=8)

    # axs[1].set_yticks([0, 15, 30])
    # axs[1].set_yticklabels(["0", "15", "30"])
    axs[0].set_yticks([0.0, 0.5, 1.0])
    axs[0].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[1].set_yticks([0.0, 0.5, 1.0])
    axs[1].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[2].set_yticks([0.0, 0.5, 1.0])
    axs[2].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[3].set_yticks([0.0, 0.5, 1.0])
    axs[3].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[4].set_yticks([0.0, 0.5, 1.0])
    axs[4].set_yticklabels(["0.0", "0.5", "1.0"])
    axs[0].spines["top"].set_visible(False)
    axs[0].spines["right"].set_visible(False)
    axs[1].spines["top"].set_visible(False)
    axs[1].spines["right"].set_visible(False)
    axs[2].spines["top"].set_visible(False)
    axs[2].spines["right"].set_visible(False)
    axs[3].spines["top"].set_visible(False)
    axs[3].spines["right"].set_visible(False)
    axs[4].spines["top"].set_visible(False)
    axs[4].spines["right"].set_visible(False)

    axs[0].legend(folder_labels, fontsize=8, loc='best', frameon=False)

    # Use same color cycle as used in the main plots
    line_colors = colors[:len(participants)]

    # Create line legend handles (matching style and colors)
    net_legend_handles = [
        Line2D([0], [0], color=color, lw=2, label=size)
        for color, size in zip(line_colors, paper_nomenclatur_dict)
    ]

    # Add legend in top-right corner, same font as reg. legend, no box
    net_legend = axs[4].legend(
        handles=net_legend_handles,
        bbox_to_anchor=(1.05, 1.2),
        title='Participant',
        fontsize=6,
        loc='upper left',
        frameon=False
    )

    # Keep reg. legend visible
    axs[4].add_artist(net_legend)

    # Show total number of models on x-axis
    axs[4].set_xlim(0, total_models)
    axs[4].set_xticks([0, total_models])
    axs[4].set_xticklabels(['0', f'{total_models}'])

    # === Save ===
    save_path = os.path.join(directory, 'visuals_overlay_robustness',
                             f'overlay_multi_density_{density}_{"-".join(folder_labels[0].split("_")[1:-1])}_{sort_variable}_{mode}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def visualize_topMarker_testPerf_corrlation(meta_perf_test_list, meta_topMarker_list, topMarker_name):
    # Args:
    # meta_perf_test_list: test performance over all grid searchs for one participant
    # meta_topMarker_list: topological marker values over all grid searchs for one participant
    # topMarker_name: topological marker name (str) - clustering, modularity, participation

    avg_perf_test_list_all_ = []
    topMarker_list_all_ = []

    avg_perf_test_list_all = []
    topMarker_list_all = []

    for s in range(0,6):
        avg_perf_test_list = list(meta_perf_test_list[s])
        topMarker_list = list(meta_topMarker_list[s])

        avg_perf_test_list_all_.extend(avg_perf_test_list)
        topMarker_list_all_.extend(topMarker_list)

    for avg_perf, topMarker in zip(
            avg_perf_test_list_all_,
            topMarker_list_all_):

        if avg_perf >= 0.22 and topMarker != 0:
            avg_perf_test_list_all.append(avg_perf)
            topMarker_list_all.append(topMarker)

    ind_sort = np.argsort(avg_perf_test_list_all)[::-1]

    perf_test_sorted = np.array(avg_perf_test_list_all)[ind_sort]
    topMarker_sorted = np.array(topMarker_list_all)[ind_sort]

    testPerfList = []
    topMarkerList = []

    # for i in range(0, 6):
    testPerfList.extend(perf_test_sorted[:])
    topMarkerList.extend(topMarker_sorted[:])

    marker = topMarker_name

    # Convert to numpy arrays
    x = np.array(topMarkerList)
    y = np.array(testPerfList)

    # Compute Pearson correlation
    r, p = pearsonr(x, y)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(4, 4))

    # Scatter plot
    ax.scatter(x, y)
    ax.set_xlabel(marker)
    ax.set_ylabel("Test Performance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Regression line
    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b)

    # Correlation annotation (inside plot)
    textstr = f"$r = {r:.2f}$\n$p = {p:.3g}$"
    ax.text(
        0.33, 0.95, textstr,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Layout and save
    fig.tight_layout()
    fig.savefig(
        fr"{directory_metaOverlayVisual}\visuals_overlay\correlation_testPerformance_{topMarker_name}_{'_'.join(foldersToOverlay[-1].split('_')[:-1])}_connectedNetworksOnly.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

def visualize_simulatedTopMarkerNetworks():
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    # =============================
    # HIGH METRICS (top row)
    # =============================

    # High clustering
    G_high_clust = nx.watts_strogatz_graph(n=40, k=6, p=0.05, seed=1)
    pos = nx.spring_layout(G_high_clust, seed=1)
    nx.draw(G_high_clust, pos, node_size=60, ax=axes[0, 0])
    axes[0, 0].set_title("High clustering")
    axes[0, 0].axis("off")

    # High modularity
    sizes = [15, 15, 10]
    probs_high_mod = [
        [0.6, 0.02, 0.02],
        [0.02, 0.6, 0.02],
        [0.02, 0.02, 0.6]
    ]
    G_high_mod = nx.stochastic_block_model(sizes, probs_high_mod, seed=2)
    pos = nx.spring_layout(G_high_mod, seed=2)
    nx.draw(G_high_mod, pos, node_size=60, ax=axes[0, 1])
    axes[0, 1].set_title("High modularity")
    axes[0, 1].axis("off")

    # High participation
    G_high_part = nx.stochastic_block_model(sizes, probs_high_mod, seed=3)
    connector = 0
    for block_start in np.cumsum([0] + sizes[:-1]):
        G_high_part.add_edge(connector, block_start)
    pos = nx.spring_layout(G_high_part, seed=3)
    nx.draw(G_high_part, pos, node_size=60, ax=axes[0, 2])
    axes[0, 2].set_title("High participation coefficient")
    axes[0, 2].axis("off")

    # =============================
    # LOW METRICS (bottom row) – CONNECTED
    # =============================

    # Low clustering but connected: random spanning tree + random edges
    G_low_clust = nx.random_tree(40, seed=4)
    extra_edges = nx.erdos_renyi_graph(40, p=0.02, seed=5)
    G_low_clust.add_edges_from(extra_edges.edges())
    pos = nx.spring_layout(G_low_clust, seed=4)
    nx.draw(G_low_clust, pos, node_size=60, ax=axes[1, 0])
    axes[1, 0].set_title("Low clustering (connected)")
    axes[1, 0].axis("off")

    # Low modularity but connected: dense random graph, ensure connectivity
    G_low_mod = nx.erdos_renyi_graph(40, p=0.12, seed=6)
    if not nx.is_connected(G_low_mod):
        for component in list(nx.connected_components(G_low_mod))[1:]:
            G_low_mod.add_edge(next(iter(component)), 0)
    pos = nx.spring_layout(G_low_mod, seed=6)
    nx.draw(G_low_mod, pos, node_size=60, ax=axes[1, 1])
    axes[1, 1].set_title("Low modularity (connected)")
    axes[1, 1].axis("off")

    # Low participation but connected: modules with minimal bridges
    probs_low_part = [
        [0.6, 0.001, 0.001],
        [0.001, 0.6, 0.001],
        [0.001, 0.001, 0.6]
    ]
    G_low_part = nx.stochastic_block_model(sizes, probs_low_part, seed=7)

    # Ensure single bridge between modules
    bridges = [(0, 15), (15, 30)]
    G_low_part.add_edges_from(bridges)

    pos = nx.spring_layout(G_low_part, seed=7)
    nx.draw(G_low_part, pos, node_size=60, ax=axes[1, 2])
    axes[1, 2].set_title("Low participation coefficient (connected)")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()

########################################################################################################################
########################################################################################################################


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

    # Definition of important variables
    foldersToOverlay = ['_gridSearch_multiTask_beRNN_03_highDimCorrects_16',
                        '_gridSearch_multiTask_beRNN_03_highDimCorrects_32',
                        '_gridSearch_multiTask_beRNN_03_highDimCorrects_64',
                        '_gridSearch_multiTask_beRNN_03_highDimCorrects_128',
                        '_gridSearch_multiTask_beRNN_03_highDimCorrects_256',
                        '_gridSearch_multiTask_beRNN_03_highDim_correctOnly_512']

    paper_nomenclatur_dict = ['HC1', 'HC2', 'MDD', 'ASD', 'SCZ']
    participantList = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05']
    participantNumber = 2 # for standard visualization
    network_sizes = ['16', '32', '64', '128', '256', '512']  # for standard visualization

    mode = ['train', 'test'][1]
    sort_variable = ['clustering', 'performance', 'silhouette'][1]

    overlay = ['standard', 'robustness'][0] # standard for one participant w. different network sizes - robustness for several participants w- one network size
    # batchPlot = True if participant == 'beRNN_03' else False
    batchPlot = [True, False][1] # info: important for robustness overlay

    lastMonth = '6'
    density = 0.1

    # Start of actual visualization
    directory_metaOverlayVisual = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__metaOverlayVisual'
    os.makedirs(directory_metaOverlayVisual, exist_ok=True)

    meta_silhouette_score_list = list()
    meta_n_clusters_list = list()
    meta_perf_train_list = list()
    meta_perf_test_list = list()
    meta_hp_list = list()
    meta_clustering_list = list()
    meta_modularity_list = list()
    meta_participation_coefficient_list = list()

    for folder in foldersToOverlay:
        final_model_dirs = []

        # standard overlay ***************************************************
        if overlay == 'standard':
            participant = participantList[participantNumber] # order unnecessary

        # robustness overlay *************************************************
        if overlay == 'robustness':
            for participant_ in participantList:
                if participant_ in folder:
                    participant = participant_
                    continue

        if 'highDim_correctOnly' in folder or 'highDimCorrects' in folder:
            dataType = 'highDim_correctOnly'
        elif 'highDim' in folder:
            dataType = 'highDim'

        directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\{folder}\{dataType}\{participant}'

        if batchPlot == False:
            model_dir_batches = os.listdir(directory)
        else:
            model_dir_batches = [folder.split('_')[-1]]

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
        n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list, avg_clustering_list, modularity_list_sparse, participation_coefficient_list = get_n_clusters(successful_model_dirs, density)
        # Store lists in meta lists for consecutive final visualization
        meta_silhouette_score_list.append(silhouette_score)
        meta_n_clusters_list.append(n_clusters)
        meta_perf_train_list.append(avg_perf_train_list)
        meta_perf_test_list.append(avg_perf_test_list)
        meta_hp_list.append(hp_list)
        meta_clustering_list.append(avg_clustering_list)
        meta_modularity_list.append(modularity_list_sparse)
        meta_participation_coefficient_list.append(participation_coefficient_list)

    # Visualize big overlay
    if overlay == 'standard':
        general_hp_plot_overlay_multiple(meta_n_clusters_list,
                                         meta_silhouette_score_list,
                                         meta_hp_list,
                                         meta_perf_train_list,
                                         meta_perf_test_list,
                                         meta_clustering_list,
                                         meta_modularity_list,
                                         meta_participation_coefficient_list,
                                         foldersToOverlay,
                                         directory_metaOverlayVisual,
                                         density,
                                         network_sizes,
                                         sort_variable='performance',
                                         mode='test',
                                         alpha=0.6,
                                         cmap_name='viridis')

        # Visualize correlation between performance and topMarker of choice
        visualize_topMarker_testPerf_corrlation(meta_perf_test_list, meta_modularity_list, 'modularity')
        # Optional for paper
        # visualize_simulatedTopMarkerNetworks()

    if overlay == 'robustness':
        participants = participantList # to overlay
        general_hp_plot_overlay_multiple_robustnessTests(meta_n_clusters_list,
                                         meta_silhouette_score_list,
                                         meta_hp_list,
                                         meta_perf_train_list,
                                         meta_perf_test_list,
                                         meta_clustering_list,
                                         meta_modularity_list,
                                         meta_participation_coefficient_list,
                                         foldersToOverlay,
                                         directory_metaOverlayVisual,
                                         density,
                                         participants,
                                         sort_variable='performance',
                                         mode='test',
                                         alpha=0.6,
                                         cmap_name='viridis')


