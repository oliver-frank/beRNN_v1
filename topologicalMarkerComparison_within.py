import json
import os
import matplotlib.pyplot as plt
import numpy as np

file = r'topologicalMarker_dict_beRNN__robustnessTest_fundamentals_participant_highDim_256_hp_2'
density = '_0.1'
folder = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists'

topMarker_dict = {
    'mod_value_sparse': 'modularity',
    'avg_clustering': 'clustering',
    'avg_eigenvector': 'eigenvector',
    'avg_betweenness': 'betweenness',
    'avg_closeness': 'closeness'
}

# load json dict
with open(os.path.join(folder, file + density + '.json'), "r") as f:
    data_dict = json.load(f)

dict_names = list(data_dict.keys())
metrics = ['mod_value_sparse', 'avg_clustering', 'avg_eigenvector', 'avg_betweenness', 'avg_closeness']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']  # consistent colors per metric

plt.figure(figsize=(12, 4))

width = 0.12  # horizontal offset for each metric within a dict
box_width = 0.1  # width of the boxplot

for i, dict_name in enumerate(dict_names):
    for j, metric in enumerate(metrics):
        y = data_dict[dict_name][metric]
        x = np.full_like(y, i + (j - 2)*width)  # scatter x positions

        # scatter plot
        plt.scatter(x, y, color=colors[j], alpha=0.7, label=topMarker_dict[metric] if i==0 else "")

        # boxplot behind scatter
        plt.boxplot(
            y,
            positions=[i + (j - 2)*width],
            widths=box_width,
            patch_artist=True,
            boxprops=dict(facecolor=colors[j], alpha=0.2, color=colors[j]),
            whiskerprops=dict(color=colors[j], alpha=0.5),
            capprops=dict(color=colors[j], alpha=0.5),
            medianprops=dict(color=colors[j])
        )

plt.xticks(range(len(dict_names)), dict_names)
plt.ylabel("Values")
title = f"groupedTopologicalMarker_{file}{density}"
plt.title(f"{title}")
plt.legend(title="Metrics", bbox_to_anchor=(1.0, 1))
plt.tight_layout()
save_path = os.path.join(folder, title + '.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()


