########################################################################################################################
# head. Track the dynamical change of internal representations (mainly topological markers) over training time #########
########################################################################################################################
# Plot train and test performance over training steps for one particular model
import json
import os
import matplotlib.pyplot as plt
from collections import OrderedDict, defaultdict
import numpy as np

participantList = ['beRNN_01','beRNN_02','beRNN_03','beRNN_04','beRNN_05']

for participant in participantList:
    # Calculate one plot for each participant with all three top markers as mean and variance over time
    if participant == 'beRNN_04':
        months = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7']
    else:
        months = ['month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7', 'month_8', 'month_9',
                  'month_10', 'month_11', 'month_12']

    data = 'highDim_correctOnly'
    base_dir_ = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\show-interDyn_multi_{participant}_{data}_256_hp1_mAll\{data}\{participant}\14'
    save_fig = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__interDynamics'
    os.makedirs(save_fig, exist_ok=True)

    topMarkerList = [
        'averageg_clustering', 'modularity_sparse', 'average_participation'
    ]

    # Load data from all months into memory first
    data_by_month_and_model = defaultdict(OrderedDict)

    for modelNumber in range(1,21):
        for month in months:
            base_dir = os.path.join(base_dir_, rf'iter{modelNumber}_LeakyRNN_diag_256_relu')
            month_path = os.path.join(base_dir, f"model_{month}", 'log.json')
            if os.path.exists(month_path):
                with open(month_path, 'r') as f:
                    data_by_month_and_model[month][modelNumber] = json.load(f)
            else:
                print(f"Warning: File not found: {month_path}")
                data_by_month_and_model[month][modelNumber] = {}

    # Setup the plot
    fig, ax = plt.subplots(figsize=(8, 3))
    colors = plt.cm.viridis(np.linspace(0, 1, len(topMarkerList)))

    stacked_y = defaultdict(OrderedDict)
    # Process and plot each task as a single consecutive timeline
    for i, marker in enumerate(topMarkerList):

        # Concatenate the lists sequentially
        for month in months:
            stacked_y_list = []

            for modelNumber in range(1,21):
                if marker in data_by_month_and_model[month][modelNumber]:
                    marker_data = data_by_month_and_model[month][modelNumber][marker]
                    stacked_y_list.append(marker_data[:27]) # should create list of lists

            stacked_y[marker][month] = stacked_y_list

            # Skip plotting if no data was found for this task across any month
            if not stacked_y:
                continue

    plt.figure(figsize=(18, 6))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    total_points = len(months) * 27
    x_axis = np.arange(total_points)

    for color_idx, (top_key, sub_dicts) in enumerate(stacked_y.items()):
        all_means = []
        all_vars = []

        # Flatten the 12 dicts sequentially
        for sub_key in sub_dicts.keys():
            # Convert the 20 lists of 27 values into a 2D NumPy array (shape: 20 x 27)
            matrix = np.array(sub_dicts[sub_key])

            # Calculate mean and variance across the 20 lists for each of the 27 positions
            means = np.mean(matrix, axis=0)
            variances = np.var(matrix, axis=0)

            all_means.extend(means)
            all_vars.extend(variances)

        # Convert to arrays for plotting
        all_means = np.array(all_means)
        all_stds = np.sqrt(np.array(all_vars))  # Standard deviation is typically preferred for error bars

        # Plot the mean line
        plt.plot(x_axis, all_means, color=colors[color_idx], label=top_key, linewidth=1.5)

        # Plot variance as a shaded area (highly recommended for 324 dense points)
        plt.fill_between(x_axis, all_means - all_stds, all_means + all_stds, color=colors[color_idx], alpha=0.15)

    # Add vertical lines to visually separate the 12 sub-dicts
    # current_boundary = 0
    month = 0
    for separator in range(27, total_points, 27):
        month += 1

        plt.axvline(x=separator, color="gray", linestyle="--", alpha=0.3)

        plt.text(separator - 13.5, 1.02, f"Month {month}",
                 color='gray', fontsize=14, ha='center', fontweight='bold',
                 transform=plt.gca().get_xaxis_transform())

    plt.text(total_points - 13.5, 1.02, f"Month {len(months)}",
             color='gray', fontsize=14, ha='center', fontweight='bold',
             transform=plt.gca().get_xaxis_transform())

    plt.title(f"{participant}", fontsize=20, y=1.1)
    plt.xlabel("Validation steps", fontsize=20)
    plt.ylabel("Marker values", fontsize=20)
    plt.ylim(0, 0.6)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim(0, 323)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Handle duplicate labels in legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(
        by_label.values(),
        ['clustering', 'modularity', 'participation'],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        fontsize=14
    )
    plt.tight_layout()

    save_path = os.path.join(save_fig, f'topMarker_overTime_{participant}_{data}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()