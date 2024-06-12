import os
# import random
import numpy as np
# import matplotlib
# matplotlib.use('WebAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'wxAgg'
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
# import tensorflow as tf

from analysis import clustering, variance
# from NETWORK import Model
import Tools
from Tools import rule_name

model_list = [
'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_145_BeRNN_05_Month_2-6'
]

# model_dir = os.getcwd() + '\BeRNN_models\Model_112_BeRNN_01_Month_2-4'
show = True # Should plots be shown or not

for model_dir in model_list:
    print(model_dir)
    ########################################################################################################################
    # Performance
    ########################################################################################################################
    _rule_color = {
                'DM': 'green',
                'DM_Anti': 'olive',
                'EF': 'forest green',
                'EF_Anti': 'mustard',
                'RP': 'tan',
                'RP_Anti': 'brown',
                'RP_Ctx1': 'lavender',
                'RP_Ctx2': 'aqua',
                'WM': 'bright purple',
                'WM_Anti': 'green blue',
                'WM_Ctx1': 'blue',
                'WM_Ctx2': 'indigo'
                }
    rule_color = {k: 'xkcd:'+v for k, v in _rule_color.items()}

    def plot_performanceprogress_eval_BeRNN(model_dir, show, rule_plot=None):
        # Plot Evaluation Progress
        log = Tools.load_log(model_dir)
        hp = Tools.load_hp(model_dir)

        # co: change to [::2] if you want to have only every second validation value
        # trials = log['trials'][::2]
        trials = log['trials']

        fs = 14 # fontsize
        fig_eval = plt.figure(figsize=(14, 6))
        ax = fig_eval.add_axes([0.1,0.4,0.6,0.5]) # co: third value influences width of cartoon
        lines = list()
        labels = list()

        x_plot = np.array(trials)/1000
        if rule_plot == None:
            rule_plot = hp['rules']

        for i, rule in enumerate(rule_plot):
            # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
            # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
            # co: add [::2] if you want to have only every second validation values
            # line = ax.plot(x_plot, np.log10(log['cost_' + 'WM'][::2]), color=rule_color[rule])
            line = ax.plot(x_plot, np.log10(log['cost_' + rule]), color=rule_color[rule])
            # co: add [::2] if you want to have only every second validation value
            # ax.plot(x_plot, log['perf_' + rule][::2], color=rule_color[rule])
            ax.plot(x_plot, log['perf_' + rule], color=rule_color[rule])
            lines.append(line[0])
            labels.append(rule_name[rule])

        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.set_ylim([0, 1])
        # ax.set_xlim([0, 80000])
        ax.set_xlabel('Total number of trials (/1000)',fontsize=fs, labelpad=2)
        ax.set_ylabel('Performance',fontsize=fs, labelpad=0)
        ax.locator_params(axis='x', nbins=5)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        lg = fig_eval.legend(lines, labels, title='Task',ncol=2,bbox_to_anchor=(0.1,0.15), # co: first value influences horizontal position of legend
                        fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        # plt.title(model_dir.split("\\")[-1]+'_EVALUATION.png') # todo: Add title
        plt.title('_'.join(model_dir.split("\\")[-1].split('_')[2:4])+'_EVALUATION',fontsize=16) # todo: Add title
        # # Add the randomness thresholds
        # # DM & RP Ctx
        # plt.axhline(y=0.2, color='green', label= 'DM & DM Anti & RP Ctx1 & RP Ctx2', linestyle=':')
        # plt.axhline(y=0.2, color='green', linestyle=':')
        # # # EF
        # plt.axhline(y=0.25, color='black', label= 'EF & EF Anti', linestyle=':')
        # plt.axhline(y=0.25, color='black', linestyle=':')
        # # # RP
        # plt.axhline(y=0.143, color='brown', label= 'RP & RP Anti', linestyle=':')
        # plt.axhline(y=0.143, color='brown', linestyle=':')
        # # # WM
        # plt.axhline(y=0.5, color='blue', label= 'WM & WM Anti & WM Ctx1 & WM Ctx2', linestyle=':')
        # plt.axhline(y=0.5, color='blue', linestyle=':')
        # #
        # rt = fig.legend(title='Randomness threshold', bbox_to_anchor=(0.1, 0.35), fontsize=fs, labelspacing=0.3  # co: first value influences length of
        #                 ,loc=6, frameon=False)
        # plt.setp(rt.get_title(), fontsize=fs)

        plt.savefig(os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\\Visuals\\Performance', model_dir.split("\\")[-1] + '_EVALUATION.png'),format='png', dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close(fig_eval)

    def plot_performanceprogress_train_BeRNN(model_dir, show, rule_plot=None):
        # Plot Training Progress
        log = Tools.load_log(model_dir)
        hp = Tools.load_hp(model_dir)

        # co: change to [::2] if you want to have only every second validation value
        # trials = log['trials'][::2]
        trials = log['trials']

        fs = 14 # fontsize
        fig_train = plt.figure(figsize=(14, 6))
        ax = fig_train.add_axes([0.1,0.4,0.6,0.5]) # co: third value influences width of cartoon
        lines = list()
        labels = list()

        x_plot = (np.array(trials)/1000)
        if rule_plot == None:
            rule_plot = hp['rules']

        for i, rule in enumerate(rule_plot):
            # line = ax1.plot(x_plot, np.log10(cost_tests[rule]),color=color_rules[i%26])
            # ax2.plot(x_plot, perf_tests[rule],color=color_rules[i%26])
            # co: add [::2] if you want to have only every second validation values
            # line = ax.plot(x_plot, np.log10(log['cost_' + 'WM'][::2]), color=rule_color[rule])
            y = log['cost_train_' + rule][::int((len(log['cost_train_' + rule])/len(x_plot)))][:len(x_plot)]
            line = ax.plot(x_plot, np.log10(y), color=rule_color[rule])
            # co: add [::2] if you want to have only every second validation value
            # ax.plot(x_plot, log['perf_' + rule][::2], color=rule_color[rule])
            y = log['perf_train_' + rule][::int((len(log['cost_train_' + rule])/len(x_plot)))][:len(x_plot)]
            ax.plot(x_plot, y, color=rule_color[rule])
            lines.append(line[0])
            labels.append(rule_name[rule])

        ax.tick_params(axis='both', which='major', labelsize=fs)

        ax.set_ylim([0, 1])
        # ax.set_xlim([0, 80000])
        ax.set_xlabel('Total number of trials (/1000)',fontsize=fs, labelpad=2)
        ax.set_ylabel('Performance',fontsize=fs, labelpad=0)
        ax.locator_params(axis='x', nbins=5)
        ax.set_yticks([0,.25,.5,.75,1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        lg = fig_train.legend(lines, labels, title='Task',ncol=2,bbox_to_anchor=(0.1,0.15), # co: first value influences horizontal position of legend
                        fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
        plt.setp(lg.get_title(),fontsize=fs)
        # plt.title(model_dir.split("\\")[-1] + '_TRAINING.png') # todo: Add title
        plt.title('_'.join(model_dir.split("\\")[-1].split('_')[2:4]) + '_TRAINING',fontsize=16)
        # # Add the randomness thresholds
        # # DM & RP Ctx
        # plt.axhline(y=0.2, color='green', label= 'DM & DM Anti & RP Ctx1 & RP Ctx2', linestyle=':')
        # plt.axhline(y=0.2, color='green', linestyle=':')
        # # # EF
        # plt.axhline(y=0.25, color='black', label= 'EF & EF Anti', linestyle=':')
        # plt.axhline(y=0.25, color='black', linestyle=':')
        # # # RP
        # plt.axhline(y=0.143, color='brown', label= 'RP & RP Anti', linestyle=':')
        # plt.axhline(y=0.143, color='brown', linestyle=':')
        # # # WM
        # plt.axhline(y=0.5, color='blue', label= 'WM & WM Anti & WM Ctx1 & WM Ctx2', linestyle=':')
        # plt.axhline(y=0.5, color='blue', linestyle=':')
        # #
        # rt = fig.legend(title='Randomness threshold', bbox_to_anchor=(0.1, 0.35), fontsize=fs, labelspacing=0.3  # co: first value influences length of
        #                 ,loc=6, frameon=False)
        # plt.setp(rt.get_title(), fontsize=fs)

        plt.savefig(os.path.join('W:\\group_csp\\analyses\\oliver.frank', 'BeRNN_models\Visuals\Performance', model_dir.split("\\")[-1] + '_TRAINING.png'),format='png',dpi=300)
        if show == True:
            plt.show()
        else:
            plt.close(fig_train)

    # Plot improvement of performance over iterating evaluation steps
    plot_performanceprogress_eval_BeRNN(model_dir, show)
    # Plot improvement of performance over iterating training steps
    plot_performanceprogress_train_BeRNN(model_dir, show)


    ########################################################################################################################
    # Clustering
    ########################################################################################################################
    # model_list = [
    #     'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_82_BeRNN_01_Month_1-4',
    #     'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_88_BeRNN_03_Month_1-4',
    #     'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_92_BeRNN_05_Month_1-4',
    # ]

    model_dir = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\' + 'Model_82_BeRNN_01_Month_1-4'
    show = True
    def compute_n_cluster(model_dir, mode, monthsConsidered, show):
    # for model_dir in model_dirs:
    #     print(model_dir)
        log = Tools.load_log(model_dir)
        hp = Tools.load_hp(model_dir)
        try:
            analysis = clustering.Analysis(model_dir, mode, monthsConsidered, 'rule')
            # Plots from instance methods in class
            # analysis.plot_cluster_score()
            analysis.plot_variance(model_dir, mode, show)
            # analysis.plot_similarity_matrix()
            analysis.plot_2Dvisualization(model_dir, mode, show)
            # analysis.plot_example_unit()
            # analysis.plot_connectivity_byclusters()

            log['n_cluster'] = analysis.n_cluster
            log['model_dir'] = model_dir
            Tools.save_log(log)
        except IOError:
            # Training never finished
            assert log['perf_min'][-1] <= hp['target_perf']

        # analysis.plot_example_unit()
        # analysis.plot_variance()
        # analysis.plot_2Dvisualization()

        print("done")

    monthsConsidered = ['1','2','3','4']
    mode = 'Training'
    compute_n_cluster(model_dir, mode, monthsConsidered, show)
    mode = 'Evaluation'
    compute_n_cluster(model_dir, mode, monthsConsidered, show)


    ########################################################################################################################
    # LAB
    ########################################################################################################################

