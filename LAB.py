def plot_aggregated_performance(model_list, mode, tasks, figure_path, month):
    """
    Plots the aggregated performance of models in training or evaluation mode across multiple tasks.
    Parameters:
        model_list (list): List of models to aggregate performance from.
        mode (str): Mode of operation, either 'train' or 'eval'.
        tasks (list): List of tasks for which to aggregate performance.
        figure_path (str): Path to save the resulting figure.
    """

    # Select the correct aggregation function based on mode
    if mode == 'train':
        aggregated_costs, aggregated_performances, x_plot = aggregate_performance_train_data(model_list, tasks)
        modus = 'Training'
    elif mode == 'eval':
        aggregated_costs, aggregated_performances, x_plot = aggregate_performance_eval_data(model_list, tasks)
        modus = 'Evaluation'

    # Create the plot
    fs = 12  # Set font size to match the second function
    fig, ax = plt.subplots(figsize=(12, 6))
    lines = []
    labels = []

    for task in tasks:
        # Convert list of arrays to a 2D array for easier mean/std calculation
        costs_array = np.array(aggregated_costs[task])
        performances_array = np.array(aggregated_performances[task])

        mean_costs = np.mean(costs_array, axis=0)
        std_costs = np.std(costs_array, axis=0)
        mean_performances = np.mean(performances_array, axis=0)
        std_performances = np.std(performances_array, axis=0)

        # Plot performance
        line, = ax.plot(x_plot, mean_performances, color=rule_color[task], linestyle='-', label=task)
        ax.fill_between(x_plot, mean_performances - std_performances, mean_performances + std_performances,
                        color=rule_color[task], alpha=0.1)

        lines.append(line)
        labels.append(rule_name[task])

    # Set labels and axis settings
    ax.tick_params(axis='both', which='major', labelsize=fs)
    ax.set_ylim([0, 1])
    ax.set_xlabel('Total number of trials (/1000)', fontsize=fs, labelpad=2)
    ax.set_ylabel('Performance', fontsize=fs, labelpad=0)
    ax.locator_params(axis='x', nbins=5)
    ax.set_yticks([0, .25, .5, .75, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # Adjust the subplot to make space for the legend below
    fig.subplots_adjust(bottom=0.3)  # Increased from 0.25 to 0.3 to create more space

    # Place the legend in a similar style to the second function but adjust its position slightly
    lg = fig.legend(lines, labels, title='Tasks', ncol=2, bbox_to_anchor=(0.5, -0.25),
                    fontsize=fs, labelspacing=0.3, loc='upper center', frameon=False)
    plt.setp(lg.get_title(), fontsize=fs)

    # Title
    subject = '_'.join(model_list[0].split("\\")[-1].split('_')[2:4])
    plt.title(f'Average {modus} Performance Across Networks - {subject} - {month}', fontsize=14)

    # Save the figure
    model_name = '_'.join(model_list[0].split("\\")[-1].split('_')[1:6])
    plt.savefig(os.path.join(figure_path, f'modelAverage_{model_name}_{modus}.png'), format='png', dpi=300)

    plt.show()
    plt.close()