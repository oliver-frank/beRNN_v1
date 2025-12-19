# Plots for topological Marker beRNN/brain comparison
import json
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# Folder containing your 45 json files
folder = r"C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists\kolmogorov\_robustnessTest_fundamentals_participant_highDim_256_hp_2"

# Regex to parse filenames like: beRNN_01_avg_closeness_0.1.json
pattern = re.compile(r"(beRNN_\d+)_(0\.\d)\.json")

rows = []

for file in os.listdir(folder):
    if file.endswith(".json"):
        match = pattern.match(file)
        if not match:
            print(f"Skipping filename: {file}")
            continue

        participant, density_str = match.groups()
        density = float(density_str)

        # load json dict
        with open(os.path.join(folder, file), "r") as f:
            data = json.load(f)

        # data contains 5 keys (your statistics)
        # we store each key-value pair separately so each JSON yields 5 rows
        for stat_name, p_value in data.items():
            rows.append({
                "participant": participant,
                "density": density,
                "statistic": stat_name,
                "p_value": p_value
            })

# Create dataframe
df = pd.DataFrame(rows)
print(df.head())



########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Unique sets
densities = sorted(df["density"].unique())             # e.g. 0.1 ... 0.9
markers = list(df["statistic"].unique())               # 5 markers
participants = list(df["participant"].unique())        # 5 participants

# Base x positions for densities
density_positions = np.arange(len(densities))          # 0...8
density_to_x = {d: density_positions[i] for i, d in enumerate(densities)}

# Sub-offsets for markers (5 per density)
marker_offsets = np.linspace(-0.25, 0.25, len(markers))
marker_to_offset = {m: marker_offsets[i] for i, m in enumerate(markers)}

# Small jitter for participants (so no overlapping)
jitter_levels = np.linspace(-0.04, 0.04, len(participants))
participant_to_jitter = {p: jitter_levels[i] for i, p in enumerate(participants)}

# Number markers 1,2,3,4,5 for participants
marker_styles = ['$1$', '$2$', '$3$', '$4$', '$5$']
participant_to_marker = {p: marker_styles[i] for i, p in enumerate(participants)}

# Color scheme: one color per statistic marker
cmap = plt.cm.get_cmap("viridis", len(markers))
marker_to_color = {m: cmap(i) for i, m in enumerate(markers)}

# -------------------------------------------------------------------
# Plot all points: BACKGROUND DOT (transparent) + NUMBER MARKER
# -------------------------------------------------------------------
for _, row in df.iterrows():

    d = row["density"]
    m = row["statistic"]
    p = row["participant"]
    pv = row["p_value"]

    # compute x position:
    base = density_to_x[d]
    lane = marker_to_offset[m]
    jitter = participant_to_jitter[p]
    x = base + lane + jitter

    c = marker_to_color[m]
    txt = participant_to_marker[p]  # '$1$' ... '$5$'

    # 1) Transparent circle in the background
    plt.scatter(
        x=x,
        y=pv,
        s=350,
        color=c,
        alpha=0.25,
        edgecolor='none',
        zorder=1
    )

    # 2) Number marker (1–5) on top
    plt.text(
        x,
        pv,
        txt,
        color=c,
        fontsize=14,
        fontweight='bold',
        ha='center',
        va='center',
        zorder=2
    )

# -------------------------------------------------------------------
# Significance line p = 0.05
# -------------------------------------------------------------------
plt.axhline(0.05, color="darkred", linestyle="--", linewidth=2, alpha=0.7, zorder=0)

# -------------------------------------------------------------------
# Legends
# -------------------------------------------------------------------
topMarker_dict = {
    'mod_value_sparse': 'modularity',
    'avg_clustering': 'clustering',
    'avg_eigenvector': 'eigenvector',
    'avg_betweenness': 'betweenness',
    'avg_closeness': 'closeness'
}
# Legend for marker types (statistic)
marker_handles = [
    plt.Line2D([0], [0], marker='o', color='w',
               markerfacecolor=marker_to_color[m], markersize=10, label=topMarker_dict[m])
    for m in markers
]

# Legend for participants (numbers 1–5)
participant_handles = [
    plt.Line2D([0], [0], marker=participant_to_marker[p], color='k',
               linestyle='None', markersize=8, label=p)
    for p in participants
]

# Place legends
first_legend = plt.legend(handles=marker_handles, title="Top. Marker",
                          bbox_to_anchor=(1.0, 1), loc="upper left")
plt.gca().add_artist(first_legend)

plt.legend(handles=participant_handles, title="Participant",
           bbox_to_anchor=(1.0, 0.55), loc="upper left")

# -------------------------------------------------------------------
# Axis formatting
# -------------------------------------------------------------------
plt.xticks(density_positions, densities)
plt.xlabel("Density")
plt.ylabel("p-value")
title = folder.split('\\')[-2:-1][0] + folder.split('\\')[-1:][0]
plt.title(f"{title}")

plt.tight_layout()
save_path = os.path.join(os.path.dirname(folder), title + '.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
