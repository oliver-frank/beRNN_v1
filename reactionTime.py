import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

# Configuration
participant_dir = r'C:\Users\oliver.frank\Desktop\PyProjects\Data'
months = ['4', '5', '6']
strToSave = months[0] + '-' + months[-1]
newParticpantList = ['beRNN_05']

# Task color dictionary
filename_color_dict = {
    'DM': '#440154',
    'DM_Anti': '#482475',
    'EF': '#31688e',
    'EF_Anti': '#26828e',
    'RP': '#35b779',
    'RP_Anti': '#6ece58',
    'RP_Ctx1': '#aadc32',
    'RP_Ctx2': '#b5de2b',
    'WM': '#fde725',
    'WM_Anti': '#fdbf11',
    'WM_Ctx1': '#ffd700',
    'WM_Ctx2': '#ffe135'
}

for participant in newParticpantList:

    fig_correct, ax_correct = plt.subplots(figsize=(10, 5))
    fig_error, ax_error = plt.subplots(figsize=(10, 5))
    fig_all, ax_all = plt.subplots(figsize=(10, 5))

    month_centers = np.arange(1, len(months) + 1)
    task_keys = list(filename_color_dict.keys())
    offsets = np.linspace(-0.4, 0.4, len(task_keys))
    box_width = 0.06

    legend_handles_correct = []
    legend_handles_error = []
    legend_handles_all = []
    seen_correct_tasks = set()
    seen_error_tasks = set()
    seen_all_tasks = set()

    stats_data = {task: {"correct": [], "error": [], "all": []} for task in task_keys}

    for m_idx, month_str in enumerate(months):
        month_center = month_centers[m_idx]
        folder = os.path.join(participant_dir, participant, month_str)

        month_files = []
        if os.path.exists(folder):
            for root, _, files in os.walk(folder):
                month_files.extend(
                    os.path.join(root, f) for f in files if f.endswith(".xlsx")
                )

        for t_idx, task in enumerate(task_keys):
            color = filename_color_dict[task]
            task_data_correct = []
            task_data_error = []
            task_data_all = []

            for filename in month_files:
                try:
                    df = pd.read_excel(filename, engine='openpyxl')

                    if (
                        isinstance(df.iloc[0, 28], str)
                        and df.iloc[0, 28].split('_trials_')[0] == task
                    ):
                        corr = (
                            df.loc[
                                (df['Component Name'] == 'Click Response') &
                                (df['Correct'] == 1),
                                'Reaction Time'
                            ]
                            .dropna()
                            .tolist()
                        )

                        err = (
                            df.loc[
                                (df['Component Name'] == 'Click Response') &
                                (df['Correct'] == 0),
                                'Reaction Time'
                            ]
                            .dropna()
                            .tolist()
                        )

                        all = (
                            df.loc[
                                (df['Component Name'] == 'Click Response'),
                                'Reaction Time'
                            ]
                            .dropna()
                            .tolist()
                        )

                        # Clean by min and max thresholds for defined reaction times - gorilla df saving errors added single values outside of range
                        if task == 'DM' or task == 'DM_Anti':
                            corr = [rt for rt in corr if rt <= 1700 and rt >= 300]
                            err = [rt for rt in err if rt <= 1700 and rt >= 300]
                            all = [rt for rt in all if rt <= 1700 and rt >= 300]
                        if task == 'EF' or task == 'EF_Anti':
                            corr = [rt for rt in corr if rt <= 900 and rt >= 300]
                            err = [rt for rt in err if rt <= 900 and rt >= 300]
                            all = [rt for rt in all if rt <= 900 and rt >= 300]
                        if task == 'RP' or task == 'RP_Anti' or task == 'RP_Ctx1' or task == 'RP_Ctx2':
                            corr = [rt for rt in corr if rt <= 1500 and rt >= 300]
                            err = [rt for rt in err if rt <= 1500 and rt >= 300]
                            all = [rt for rt in all if rt <= 1500 and rt >= 300]
                        if task == 'WM' or task == 'WM_Anti' or task == 'WM_Ctx1' or task == 'WM_Ctx2':
                            corr = [rt for rt in corr if rt <= 1500 and rt >= 300]
                            err = [rt for rt in err if rt <= 1500 and rt >= 300]
                            all = [rt for rt in all if rt <= 1500 and rt >= 300]

                        task_data_correct.extend(corr)
                        task_data_error.extend(err)
                        task_data_all.extend(all)

                        stats_data[task]["correct"].extend(corr)
                        stats_data[task]["error"].extend(err)
                        stats_data[task]["all"].extend(all)

                except Exception as e:
                    print(f"Error in {filename}: {e}")



            # ---- Correct plots ----
            if task_data_correct:
                x_pos = month_center + offsets[t_idx]

                ax_correct.boxplot(
                    task_data_correct,
                    positions=[x_pos],
                    widths=box_width,
                    patch_artist=True,
                    showmeans=True,
                    showfliers=False,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=color, markersize=4),
                    boxprops=dict(facecolor=color, color=color, alpha=0.5),
                    medianprops=dict(color="white")
                )

                jitter = np.random.uniform(-box_width / 4, box_width / 4,
                                           size=len(task_data_correct))
                ax_correct.scatter(
                    np.full(len(task_data_correct), x_pos) + jitter,
                    task_data_correct,
                    color=color, alpha=0.3, s=8
                )

                if task not in seen_correct_tasks:
                    legend_handles_correct.append(
                        plt.Line2D([0], [0], color=color, lw=4, label=task)
                    )
                    seen_correct_tasks.add(task)



            # ---- Error plots ----
            if task_data_error:
                x_pos = month_center + offsets[t_idx]

                ax_error.boxplot(
                    task_data_error,
                    positions=[x_pos],
                    widths=box_width,
                    patch_artist=True,
                    showmeans=True,
                    showfliers=False,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=color, markersize=4),
                    boxprops=dict(facecolor=color, color=color, alpha=0.5),
                    medianprops=dict(color="white")
                )

                jitter = np.random.uniform(-box_width / 4, box_width / 4,
                                           size=len(task_data_error))
                ax_error.scatter(
                    np.full(len(task_data_error), x_pos) + jitter,
                    task_data_error,
                    color=color, alpha=0.3, s=8
                )

                # ---- All plots ----
                if task not in seen_error_tasks:
                    legend_handles_error.append(
                        plt.Line2D([0], [0], color=color, lw=4, label=task)
                    )
                    seen_error_tasks.add(task)



            # ---- Error plots ----
            if task_data_all:
                x_pos = month_center + offsets[t_idx]

                ax_all.boxplot(
                    task_data_all,
                    positions=[x_pos],
                    widths=box_width,
                    patch_artist=True,
                    showmeans=True,
                    showfliers=False,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor=color, markersize=4),
                    boxprops=dict(facecolor=color, color=color, alpha=0.5),
                    medianprops=dict(color="white")
                )

                jitter = np.random.uniform(-box_width / 4, box_width / 4,
                                           size=len(task_data_all))
                ax_all.scatter(
                    np.full(len(task_data_all), x_pos) + jitter,
                    task_data_all,
                    color=color, alpha=0.3, s=8
                )

                if task not in seen_all_tasks:
                    legend_handles_all.append(
                        plt.Line2D([0], [0], color=color, lw=4, label=task)
                    )
                    seen_all_tasks.add(task)


    # ---------- Formatting ----------
    for ax in (ax_correct, ax_error, ax_all):
        ax.set_ylim(0, 1500)
        ax.set_yticks(range(0, 1501, 500))
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.set_xticks(month_centers)
        ax.set_xticklabels([f"Month {m}" for m in months])
        ax.set_xlim(month_centers[0] - 0.5, month_centers[-1] + 0.5)
        ax.set_ylabel("Reaction Time (ms)")

    ax_correct.set_title(f"Reaction Time – Correct ({participant})")
    ax_error.set_title(f"Reaction Time – Errors ({participant})")
    ax_all.set_title(f"Reaction Time – All ({participant})")

    ax_correct.legend(handles=legend_handles_correct,
                      loc='upper left', bbox_to_anchor=(1.02, 1))
    ax_error.legend(handles=legend_handles_error,
                    loc='upper left', bbox_to_anchor=(1.02, 1))
    ax_all.legend(handles=legend_handles_all,
                    loc='upper left', bbox_to_anchor=(1.02, 1))

    fig_correct.subplots_adjust(right=0.80)
    fig_error.subplots_adjust(right=0.80)
    fig_all.subplots_adjust(right=0.80)

    fig_correct.savefig(
        os.path.join(participant_dir, participant,
                     f"{participant}_{strToSave}_reactionTime_Corrects.png"),
        dpi=300
    )
    fig_error.savefig(
        os.path.join(participant_dir, participant,
                     f"{participant}_{strToSave}_reactionTime_Errors.png"),
        dpi=300
    )
    fig_all.savefig(
        os.path.join(participant_dir, participant,
                     f"{participant}_{strToSave}_reactionTime_All.png"),
        dpi=300
    )

    plt.close(fig_correct)
    plt.close(fig_error)
    plt.close(fig_all)



    # ---------- Statistics ----------
    print(f"\n{'=' * 95}")
    print(f"STATISTICAL ANALYSIS: Correct vs. Error Reaction Times ({participant})")
    print(f"{'=' * 95}")

    # Define thresholds
    alpha_standard = 0.05
    n_tests = len([t for t in stats_data if len(stats_data[t]["correct"]) >= 10 and len(stats_data[t]["error"]) >= 10])
    # Protect against division by zero if no tasks have enough data
    alpha_bonf = alpha_standard / max(n_tests, 1)

    print(
        f"Significance Thresholds: Standard Alpha = {alpha_standard:.2f} | Bonferroni Alpha (n={n_tests}) = {alpha_bonf:.4f}")
    print(f"{'-' * 95}")
    print(
        f"{'Task':<10} | {'n_corr':<7} | {'n_err':<7} | {'p-value':<10} | {'Effect (r)':<10} | {'Sig (0.05)':<10} | {'Sig (Bonf)':<10}")
    print(f"{'-' * 95}")

    for task in task_keys:
        d_corr = stats_data[task]["correct"]
        d_err = stats_data[task]["error"]
        n1, n2 = len(d_corr), len(d_err)

        if n1 >= 10 and n2 >= 10:
            # Perform Mann-Whitney U
            u_stat, p = mannwhitneyu(d_corr, d_err, alternative="two-sided")

            # Calculate Effect Size r = |Z| / sqrt(N)
            # Standardizing U to Z
            mu_u = (n1 * n2) / 2
            sigma_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            # Avoid division by zero in edge cases
            z = (u_stat - mu_u) / sigma_u if sigma_u != 0 else 0
            effect_size_r = abs(z) / np.sqrt(n1 + n2)

            # Determine significance strings
            sig_std = "YES" if p < alpha_standard else "no"
            sig_bonf = "YES" if p < alpha_bonf else "no"

            print(
                f"{task:<10} | {n1:<7} | {n2:<7} | {p:<10.2e} | {effect_size_r:<10.3f} | {sig_std:<10} | {sig_bonf:<10}")
        else:
            # Handle cases with insufficient data
            reason = f"n={n1}/{n2}"
            print(f"{task:<10} | {reason:<30} | Insufficient data for reliable test")

    print(f"{'=' * 95}\n")
