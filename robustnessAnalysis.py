########################################################################################################################
# head: Robustness analysis
########################################################################################################################

# Compare silhouette, number of clusters, test performance and training performance distributions of 20 best models with
# robustnessTest distribution (20 models) of each model respectively
from hyperparameterOverview import get_n_clusters, compute_n_cluster
import os

participant = ['beRNN_01', 'beRNN_02', 'beRNN_03', 'beRNN_04', 'beRNN_05'][2]
dataType = ['highDim', 'highDim_3stimTC', 'highDim_correctOnly'][2]

mode = ['train', 'test'][1]
# sort_variable = ['clustering', 'performance', 'silhouette'][1]
batchPlot, batch = [True, False][0], '0'
lastMonth = '6'

directory = fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\robustnessTest\{dataType}\{participant}'
model_dir_batches = [mdb for mdb in os.listdir(directory) if mdb != 'visuals']

robustnessDict = {
    'silhouette_scores': [],
    'n_clusters_scores': [],
    'avg_perf_train_scores': [],
    'avg_perf_test_scores': []
}

# info: Create one distribution for each model representing the variance of the robustnessTest
for model_dir_batch in model_dir_batches:
    model_dir_group_ = [mdg for mdg in os.listdir(os.path.join(directory, model_dir_batch)) if not mdg.endswith('.txt')]
    final_model_dirs = []

    for model_dir_group in model_dir_group_:
        model_dirs_ = os.listdir(os.path.join(directory, model_dir_batch, model_dir_group))
        model_dirs = [model_dir for model_dir in model_dirs_ if model_dir != 'overviews' and not model_dir.endswith('.txt')]
        for model_dir in model_dirs:
            # Concatenate all models in one list
            if 'model' and lastMonth in model_dir: # Be sure to add anything else but models
                final_model_dirs.append(os.path.join(directory, model_dir_batch, model_dir_group, model_dir))

    print(len(final_model_dirs))
    # Compute all values for model group of current batch
    successful_model_dirs = compute_n_cluster(final_model_dirs, mode)
    n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list = get_n_clusters(successful_model_dirs)

    robustnessDict['silhouette_scores'].append(silhouette_score)
    robustnessDict['n_clusters_scores'].append(n_clusters)
    robustnessDict['avg_perf_train_scores'].append(avg_perf_train_list)
    robustnessDict['avg_perf_test_scores'].append(avg_perf_test_list)


# info: Create one distribution representing x best models once
model_dir_batches = [mdb for mdb in os.listdir(directory) if mdb != 'visuals']
final_model_dirs = []

for model_dir_batch in model_dir_batches:
    model_dir = os.path.join(directory, model_dir_batch, os.listdir(os.path.join(directory, model_dir_batch))[0], f'model_month_{lastMonth}')
    final_model_dirs.append(model_dir)

# Compute all values for model group of current batch
successful_model_dirs = compute_n_cluster(final_model_dirs, mode)
n_clusters, hp_list, silhouette_score, avg_perf_train_list, avg_perf_test_list = get_n_clusters(successful_model_dirs)

# info: [-1] list is always the general representative distribution to compare all other distributions with
robustnessDict['silhouette_scores'].append(silhouette_score)
robustnessDict['n_clusters_scores'].append(n_clusters)
robustnessDict['avg_perf_train_scores'].append(avg_perf_train_list)
robustnessDict['avg_perf_test_scores'].append(avg_perf_test_list)



# Compare x best model representation with each modelGroup distribution
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp  # or use mannwhitneyu
import numpy as np

def robustnessPlots(robustnessVariable, distributions, directory):
    n = len(distributions)
    fig, axes = plt.subplots(n, n, figsize=(n, n))
    fig.suptitle(f"{robustnessVariable}", fontsize=20, y=0.99)

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Plot histogram on the diagonal
                ax.hist(distributions[i], bins=20, color='gray', alpha=0.7)
                ax.set_facecolor('white')
                # Compute and annotate standard deviation
                std = np.std(distributions[i])
                ax.text(
                    0.05, 0.9,
                    f"std={std:.2f}",
                    ha='left', va='center', fontsize=10,
                    transform=ax.transAxes, color='black'
                )
            else:
                # Scatter plot off-diagonal
                ax.scatter(distributions[j], distributions[i], alpha=0.4, s=10)

                # Statistical test if samples come from same distribution
                stat, p = ks_2samp(distributions[i], distributions[j])

                # If significant (e.g., p < 0.05), highlight cell
                if p < 0.05:
                    ax.set_facecolor('#ffe6e6')  # Light red for significance

                # Annotate p-value
                ax.text(
                    0.5, 0.85,
                    f"p={p:.3f}",
                    ha='center', va='center', fontsize=10,
                    transform=ax.transAxes
                )

            ax.set_xticks([])
            ax.set_yticks([])
            if i == n - 1:
                ax.set_xlabel(f"Dist {j + 1}", fontsize=15)
            if j == 0:
                ax.set_ylabel(f"Dist {i + 1}", fontsize=15)

    plt.tight_layout()

    saveDirectory = os.path.join(directory, 'visuals', 'robustnessPlots')
    os.makedirs(saveDirectory, exist_ok=True)
    plt.savefig(os.path.join(saveDirectory, f'{robustnessVariable}_plot.png'))
    # plt.show()

keys = [key for key in robustnessDict.keys()]
# Create plot for each gathered robustness variable in robustnessDict - last distribution represent general representation of x best models
for key in keys:
    robustnessPlots(key, robustnessDict[key], directory)



# fix Create modularity over training time plots for robustness model batch
# fix Create performance over training time plots for robustness model batch


