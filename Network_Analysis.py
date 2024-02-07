import os
import random
import numpy as np
import matplotlib
# matplotlib.use('WebAgg')  # Or 'Qt5Agg', 'GTK3Agg', 'wxAgg'
import matplotlib.pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

from analysis import clustering #, variance
from NETWORK import Model
import TOOLS
from TOOLS import rule_name

# co: It doesn't work because somewehere in the pipeline the path is set to OLD folder in BeRNN models, change that
# model_dir = os.getcwd() + '\BeRNN_models\OLD\MH_500_train-err_validate-acc'
# variance.compute_variance(model_dir, random_rotation='random_rotation')

########################################################################################################################
'''Network analysis'''
########################################################################################################################
# Analysis functions
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

# def easy_activity_plot_BeRNN(model_dir, rule):
#     """A simple plot of neural activity from one task.
#
#     Args:
#         model_dir: directory where model file is saved
#         rule: string, the rule to plot
#     """
#
#     model = Model(model_dir)
#     hp = model.hp
#
#     # todo: Create taskList to generate trials from
#     # co: If you want to test a general model: Put all relevant .xlsx files in one folder so that the pipeline still works
#     xlsxFolder = os.getcwd() + '\\Data CSP\\MH\\'
#     AllTasks_list = fileDict_error(xlsxFolder)
#     random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))
#
#
#     with tf.Session() as sess:
#         model.restore()
#
#         currentRule = ' '
#         while currentRule != rule:
#             currentBatch = random.sample(random_AllTasks_list, 1)[0]
#             if len(currentBatch.split('_')) == 6:
#                 currentRule = currentBatch.split('_')[2] + ' ' + currentBatch.split('_')[3]
#             else:
#                 currentRule = currentBatch.split('_')[2]
#
#         if currentBatch.split('_')[2] == 'DM':
#             Input, Output, y_loc, epochs = prepare_DM_error(currentBatch, 48, 60)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
#         elif currentBatch.split('_')[2] == 'EF':
#             Input, Output, y_loc, epochs = prepare_EF_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'RP':
#             Input, Output, y_loc, epochs = prepare_RP_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'WM':
#             Input, Output, y_loc, epochs = prepare_WM_error(currentBatch, 48, 60)
#
#
#         feed_dict = Tools.gen_feed_dict_BeRNN(model, Input, Output, hp)
#         h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
#         # All matrices have shape (n_time, n_condition, n_neuron)
#
#     # Take only the one example trial
#     i_trial = 10
#
#     for activity, title in zip([Input, h, y_hat],['input', 'recurrent', 'output']):
#         plt.figure()
#         plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot',      # np.uint8
#                    interpolation='none', origin='lower')
#         plt.title(title)
#         plt.colorbar()
#
#         plt.ioff()
#         plt.show()

# Bug fixing: model_dir = os.getcwd() + '\BeRNN_models\Model_BeRNN_01_Month_1-2'
def plot_performanceprogress_BeRNN(model_dir, rule_plot=None):
    # Plot Training Progress
    log = TOOLS.load_log(model_dir)
    hp = TOOLS.load_hp(model_dir)

    # co: change to [::2] if you want to have only every second validation value
    # trials = log['trials'][::2]
    trials = log['trials']

    fs = 12 # fontsize
    fig = plt.figure(figsize=(3.5,1.2))
    ax = fig.add_axes([0.1,0.4,0.6,0.5]) # co: third value influences width of cartoon
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
    ax.set_yticks([0,1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    lg = fig.legend(lines, labels, title='Task',ncol=2,bbox_to_anchor=(0.1,0.15), # co: first value influences horizontal position of legend
                    fontsize=fs,labelspacing=0.3,loc=6,frameon=False)
    plt.setp(lg.get_title(),fontsize=fs)
    # # Add the randomness thresholds
    # # DM & RP Ctx
    plt.axhline(y=0.2, color='green', label= 'DM & DM Anti & RP Ctx1 & RP Ctx2', linestyle=':')
    plt.axhline(y=0.2, color='green', linestyle=':')
    # # EF
    plt.axhline(y=0.25, color='black', label= 'EF & EF Anti', linestyle=':')
    plt.axhline(y=0.25, color='black', linestyle=':')
    # # RP
    plt.axhline(y=0.143, color='brown', label= 'RP & RP Anti', linestyle=':')
    plt.axhline(y=0.143, color='brown', linestyle=':')
    # # WM
    plt.axhline(y=0.5, color='blue', label= 'WM & WM Anti & WM Ctx1 & WM Ctx2', linestyle=':')
    plt.axhline(y=0.5, color='blue', linestyle=':')
    # #
    # rt = fig.legend(title='Randomness threshold', bbox_to_anchor=(0.1, 0.35), fontsize=fs, labelspacing=0.3  # co: first value influences length of
    #                 ,loc=6, frameon=False)
    # plt.setp(rt.get_title(), fontsize=fs)

    plt.show()

# Input (>300?) to Hidden; Hidden to Output
# def easy_connectivity_plot_BeRNN(model_dir):
#     """A simple plot of network connectivity."""
#
#     model = Model(model_dir)
#     with tf.Session() as sess:
#         model.restore()
#         # get all connection weights and biases as tensorflow variables
#         var_list = model.var_list
#         # evaluate the parameters after training
#         params = [sess.run(var) for var in var_list]
#         # get name of each variable
#         names = [var.name for var in var_list]
#
#     # Plot weights
#     for param, name in zip(params, names):
#         if len(param.shape) != 2:
#             continue
#
#         vmax = np.max(abs(param)) * 0.7
#         plt.figure()
#         # notice the transpose
#         plt.imshow(param.T, aspect='auto', cmap='bwr', vmin=-vmax, vmax=vmax,
#                    interpolation='none', origin='lower')
#         plt.title(name)
#         plt.colorbar()
#         plt.xlabel('From')
#         plt.ylabel('To')
#         plt.show()#

# def pretty_inputoutput_plot_BeRNN(model_dir, rule, save=False, plot_ylabel=False):
#     """Plot the input and output activity for a sample trial from one task.
#
#     Args:
#         model_dir: model directory
#         rule: string, the rule
#         save: bool, whether to save plots
#         plot_ylabel: bool, whether to plot ylable
#     """
#
#     fs = 7
#
#     model = Model(model_dir)
#     hp = model.hp
#
#     # todo: Create taskList to generate trials from ####################################################################
#     # co: If you want to test a general model: Put all relevant .xlsx files in one folder so that the pipeline still works
#     xlsxFolder = os.getcwd() + '\\Data CSP\\MH\\'
#     AllTasks_list = fileDict_error(xlsxFolder)
#     random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))
#
#     with tf.Session() as sess:
#         model.restore()
#
#         currentRule = ' '
#         while currentRule != rule:
#             currentBatch = random.sample(random_AllTasks_list, 1)[0]
#             if len(currentBatch.split('_')) == 6:
#                 currentRule = currentBatch.split('_')[2] + ' ' + currentBatch.split('_')[3]
#             else:
#                 currentRule = currentBatch.split('_')[2]
#
#         if currentBatch.split('_')[2] == 'DM':
#             Input, Output, y_loc, epochs = prepare_DM_error(currentBatch, 48, 60)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
#         elif currentBatch.split('_')[2] == 'EF':
#             Input, Output, y_loc, epochs = prepare_EF_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'RP':
#             Input, Output, y_loc, epochs = prepare_RP_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'WM':
#             Input, Output, y_loc, epochs = prepare_WM_error(currentBatch, 48, 60)
#
#         feed_dict = Tools.gen_feed_dict_BeRNN(model, Input, Output, hp)
#         h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
#         # todo: ########################################################################################################
#         t_plot = np.arange(Input.shape[0]) * hp['dt'] / 1000
#
#         assert hp['num_ring'] == 2
#
#         n_eachring = hp['n_eachring']
#
#         fig = plt.figure(figsize=(1.3, 2))
#         ylabels = ['fix. in', 'stim. mod1', 'stim. mod2', 'fix. out', 'out']
#         heights = np.array([0.03, 0.2, 0.2, 0.03, 0.2]) + 0.01
#         for i in range(5):
#             ax = fig.add_axes([0.15, sum(heights[i + 1:] + 0.02) + 0.1, 0.8, heights[i]])
#             cmap = 'Purples'
#             plt.xticks([])
#             ax.tick_params(axis='both', which='major', labelsize=fs,
#                            width=0.5, length=2, pad=3)
#
#             if plot_ylabel:
#                 ax.spines["right"].set_visible(False)
#                 ax.spines["bottom"].set_visible(False)
#                 ax.spines["top"].set_visible(False)
#                 ax.xaxis.set_ticks_position('bottom')
#                 ax.yaxis.set_ticks_position('left')
#
#             else:
#                 ax.spines["left"].set_visible(False)
#                 ax.spines["right"].set_visible(False)
#                 ax.spines["bottom"].set_visible(False)
#                 ax.spines["top"].set_visible(False)
#                 ax.xaxis.set_ticks_position('none')
#
#             if i == 0:
#                 plt.plot(t_plot, Input[:, 0, 0], color='xkcd:blue')
#                 if plot_ylabel:
#                     plt.yticks([0, 1], ['', ''], rotation='vertical')
#                 plt.ylim([-0.1, 1.5])
#                 plt.title(rule_name[rule], fontsize=fs)
#             elif i == 1:
#                 plt.imshow(Input[:, 0, 1:1 + n_eachring].T, aspect='auto', cmap=cmap,
#                            vmin=0, vmax=1, interpolation='none', origin='lower')
#                 if plot_ylabel:
#                     plt.yticks([0, (n_eachring - 1) / 2, n_eachring - 1],
#                                [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
#                                rotation='vertical')
#             elif i == 2:
#                 plt.imshow(Input[:, 0, 1 + n_eachring:1 + 2 * n_eachring].T,
#                            aspect='auto', cmap=cmap, vmin=0, vmax=1,
#                            interpolation='none', origin='lower')
#
#                 if plot_ylabel:
#                     plt.yticks(
#                         [0, (n_eachring - 1) / 2, n_eachring - 1],
#                         [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
#                         rotation='vertical')
#             elif i == 3:
#                 plt.plot(t_plot, Output[:, 0, 0], color='xkcd:green')
#                 plt.plot(t_plot, y_hat[:, 0, 0], color='xkcd:blue')
#                 if plot_ylabel:
#                     plt.yticks([0.05, 0.8], ['', ''], rotation='vertical')
#                 plt.ylim([-0.1, 1.1])
#             elif i == 4:
#                 plt.imshow(y_hat[:, 0, 1:].T, aspect='auto', cmap=cmap,
#                            vmin=0, vmax=1, interpolation='none', origin='lower')
#                 if plot_ylabel:
#                     plt.yticks(
#                         [0, (n_eachring - 1) / 2, n_eachring - 1],
#                         [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
#                         rotation='vertical')
#                 plt.xticks([0, y_hat.shape[0]], ['0', '2'])
#                 plt.xlabel('Time (s)', fontsize=fs, labelpad=-3)
#                 ax.spines["bottom"].set_visible(True)
#
#             if plot_ylabel:
#                 plt.ylabel(ylabels[i], fontsize=fs)
#             else:
#                 plt.yticks([])
#             ax.get_yaxis().set_label_coords(-0.12, 0.5)
#
#         if save:
#             save_name = 'figure/sample_' + rule_name[rule].replace(' ', '') + '.pdf'
#             plt.savefig(save_name, transparent=True)
#         plt.show()
#
#         # plt.figure()
#         # _ = plt.plot(h_sample[:,0,:20])
#         # plt.show()
#         #
#         # plt.figure()
#         # _ = plt.plot(y_sample[:,0,:])
#         # plt.show()

# def schematic_plot_BeRNN(model_dir, rule=None):
#     fontsize = 6
#
#     rule = rule or 'DM'
#
#     model = Model(model_dir, dt=1)
#     hp = model.hp
#
#     with tf.Session() as sess:
#         model.restore()
#
#         # todo: Create taskList to generate trials from ####################################################################
#         # co: If you want to test a general model: Put all relevant .xlsx files in one folder so that the pipeline still works
#         xlsxFolder = os.getcwd() + '\\Data CSP\\MH\\'
#         AllTasks_list = fileDict_error(xlsxFolder)
#         random_AllTasks_list = random.sample(AllTasks_list, len(AllTasks_list))
#
#         currentRule = ' '
#         while currentRule != rule:
#             currentBatch = random.sample(random_AllTasks_list, 1)[0]
#             if len(currentBatch.split('_')) == 6:
#                 currentRule = currentBatch.split('_')[2] + ' ' + currentBatch.split('_')[3]
#             else:
#                 currentRule = currentBatch.split('_')[2]
#
#         if currentBatch.split('_')[2] == 'DM':
#             Input, Output, y_loc, epochs = prepare_DM_error(currentBatch, 48, 60)  # co: cmask problem: (model, hp['loss_type'], currentBatch, 0, 48)
#         elif currentBatch.split('_')[2] == 'EF':
#             Input, Output, y_loc, epochs = prepare_EF_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'RP':
#             Input, Output, y_loc, epochs = prepare_RP_error(currentBatch, 48, 60)
#         elif currentBatch.split('_')[2] == 'WM':
#             Input, Output, y_loc, epochs = prepare_WM_error(currentBatch, 48, 60)
#
#         feed_dict = Tools.gen_feed_dict_BeRNN(model, Input, Output, hp)
#         h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
#         # todo: ########################################################################################################
#
#
#     n_eachring = hp['n_eachring']
#     n_hidden = hp['n_rnn']
#
#     # Plot Stimulus
#     fig = plt.figure(figsize=(1.0,1.2))
#     heights = np.array([0.06,0.25,0.25])
#     for i in range(3):
#         ax = fig.add_axes([0.2,sum(heights[i+1:]+0.1)+0.05,0.7,heights[i]])
#         cmap = 'Purples'
#         plt.xticks([])
#
#         # Fixed style for these plots
#         ax.tick_params(axis='both', which='major', labelsize=fontsize, width=0.5, length=2, pad=3)
#         ax.spines["left"].set_linewidth(0.5)
#         ax.spines["right"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.xaxis.set_ticks_position('bottom')
#         ax.yaxis.set_ticks_position('left')
#
#         if i == 0:
#             plt.plot(Input[:,0,0], color='xkcd:blue')
#             plt.yticks([0, 1], ['', ''],rotation='vertical')
#             plt.ylim([-0.1, 1.5])
#             plt.title('Fixation input', fontsize=fontsize, y=0.9)
#         elif i == 1:
#             plt.imshow(Input[:, 0, 1:1+n_eachring].T, aspect='auto', cmap=cmap,
#                        vmin=0, vmax=1, interpolation='none',origin='lower')
#             plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
#                        [r'0$\degree$', '', r'360$\degree$'],
#                        rotation='vertical')
#             plt.title('Stimulus mod 1', fontsize=fontsize, y=0.9)
#         elif i == 2:
#             plt.imshow(Input[:, 0, 1+n_eachring:1+2*n_eachring].T, aspect='auto',
#                        cmap=cmap, vmin=0, vmax=1,
#                        interpolation='none', origin='lower')
#             plt.yticks([0, (n_eachring-1)/2, n_eachring-1], ['', '', ''],
#                        rotation='vertical')
#             plt.title('Stimulus mod 2', fontsize=fontsize, y=0.9)
#         ax.get_yaxis().set_label_coords(-0.12,0.5)
#     # plt.savefig('figure/schematic_input.pdf',transparent=True)
#     plt.show()
#
#     # Plot Rule Inputs
#     fig = plt.figure(figsize=(1.0, 0.5))
#     ax = fig.add_axes([0.2,0.3,0.7,0.45])
#     cmap = 'Purples'
#     X = Input[:, 0, 1+2*n_eachring:]
#     plt.imshow(X.T, aspect='auto', vmin=0, vmax=1, cmap=cmap,
#                interpolation='none', origin='lower')
#
#     plt.xticks([0, X.shape[0]])
#     ax.set_xlabel('Time (ms)', fontsize=fontsize, labelpad=-5)
#
#     # Fixed style for these plots
#     ax.tick_params(axis='both', which='major', labelsize=fontsize,
#                    width=0.5, length=2, pad=3)
#     ax.spines["left"].set_linewidth(0.5)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_linewidth(0.5)
#     ax.spines["top"].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#
#     plt.yticks([0, X.shape[-1]-1], ['1',str(X.shape[-1])], rotation='vertical')
#     plt.title('Rule inputs', fontsize=fontsize, y=0.9)
#     ax.get_yaxis().set_label_coords(-0.12,0.5)
#
#     # plt.savefig('figure/schematic_rule.pdf',transparent=True)
#     plt.show()
#
#
#     # Plot Units
#     fig = plt.figure(figsize=(1.0, 0.8))
#     ax = fig.add_axes([0.2,0.1,0.7,0.75])
#     cmap = 'Purples'
#     plt.xticks([])
#     # Fixed style for these plots
#     ax.tick_params(axis='both', which='major', labelsize=fontsize,
#                    width=0.5, length=2, pad=3)
#     ax.spines["left"].set_linewidth(0.5)
#     ax.spines["right"].set_visible(False)
#     ax.spines["bottom"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.xaxis.set_ticks_position('bottom')
#     ax.yaxis.set_ticks_position('left')
#
#     plt.imshow(h[:, 0, :].T, aspect='auto', cmap=cmap, vmin=0, vmax=1,
#                interpolation='none',origin='lower')
#     plt.yticks([0,n_hidden-1],['1',str(n_hidden)],rotation='vertical')
#     plt.title('Recurrent units', fontsize=fontsize, y=0.95)
#     ax.get_yaxis().set_label_coords(-0.12,0.5)
#     # plt.savefig('figure/schematic_units.pdf',transparent=True)
#     plt.show()
#
#
#     # Plot Outputs
#     fig = plt.figure(figsize=(1.0,0.8))
#     heights = np.array([0.1,0.45])+0.01
#     for i in range(2):
#         ax = fig.add_axes([0.2, sum(heights[i+1:]+0.15)+0.1, 0.7, heights[i]])
#         cmap = 'Purples'
#         plt.xticks([])
#
#         # Fixed style for these plots
#         ax.tick_params(axis='both', which='major', labelsize=fontsize,
#                        width=0.5, length=2, pad=3)
#         ax.spines["left"].set_linewidth(0.5)
#         ax.spines["right"].set_visible(False)
#         ax.spines["bottom"].set_visible(False)
#         ax.spines["top"].set_visible(False)
#         ax.xaxis.set_ticks_position('bottom')
#         ax.yaxis.set_ticks_position('left')
#
#         if i == 0:
#             plt.plot(y_hat[:,0,0],color='xkcd:blue')
#             plt.yticks([0.05,0.8],['',''],rotation='vertical')
#             plt.ylim([-0.1,1.1])
#             plt.title('Fixation output', fontsize=fontsize, y=0.9)
#
#         elif i == 1:
#             plt.imshow(y_hat[:,0,1:].T, aspect='auto', cmap=cmap,
#                        vmin=0, vmax=1, interpolation='none', origin='lower')
#             plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
#                        [r'0$\degree$', '', r'360$\degree$'],
#                        rotation='vertical')
#             plt.xticks([])
#             plt.title('Response', fontsize=fontsize, y=0.9)
#
#         ax.get_yaxis().set_label_coords(-0.12,0.5)
#
#     # plt.savefig('figure/schematic_outputs.pdf',transparent=True)
#     plt.show()

# def networkx_illustration_BeRNN(model_dir):
#     import networkx as nx
#
#     model = Model(model_dir)
#     with tf.Session() as sess:
#         model.restore()
#         # get all connection weights and biases as tensorflow variables
#         w_rec = sess.run(model.w_rec)
#
#     w_rec_flat = w_rec.flatten()
#     ind_sort = np.argsort(abs(w_rec_flat - np.mean(w_rec_flat)))
#     n_show = int(0.01 * len(w_rec_flat))
#     ind_gone = ind_sort[:-n_show]
#     ind_keep = ind_sort[-n_show:]
#     w_rec_flat[ind_gone] = 0
#     w_rec2 = np.reshape(w_rec_flat, w_rec.shape)
#     w_rec_keep = w_rec_flat[ind_keep]
#     G = nx.from_numpy_array(abs(w_rec2), create_using=nx.DiGraph())
#
#     color = w_rec_keep
#     fig = plt.figure(figsize=(4, 4))
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     nx.draw(G,
#             linewidths=0,
#             width=0.1,
#             alpha=1.0,
#             edge_vmin=-3,
#             edge_vmax=3,
#             arrows=False,
#             pos=nx.circular_layout(G),
#             node_color=np.array([99. / 255] * 3),
#             node_size=10,
#             edge_color=color,
#             edge_cmap=plt.cm.RdBu_r,
#             ax=ax)
#     # plt.savefig('figure/illustration_networkx.pdf', transparent=True)



# todo: ################################################################################################################
# todo: ################################################################################################################
model_dir = os.getcwd() + '\BeRNN_models\Model_14_BeRNN_03_Month_1-2'
# rule = 'WM'
# Plot activity of input, recurrent and output layer for one test trial
# easy_activity_plot_BeRNN(model_dir, rule)
# Plot improvement of performance over iterating training steps
plot_performanceprogress_BeRNN(model_dir)
plt.savefig(os.path.join(os.getcwd(),'BeRNN_models\Visuals',model_dir.split("\\")[-1]+'.png'), format='png')
# easy_connectivity_plot_BeRNN(model_dir)
#
# pretty_inputoutput_plot_BeRNN(model_dir,rule)

# schematic_plot_BeRNN(model_dir, rule)

# networkx_illustration_BeRNN(model_dir)


#######################################################################################################################
# Clustering
#######################################################################################################################
# model_dir = os.getcwd() + '\BeRNN_models\Model_BeRNN_02_Month_1-2'
# # model_dir = os.getcwd() + '\BeRNN_models\OLD\MH_200_train-err_validate-err'
# def compute_n_cluster(model_dir):
# # for model_dir in model_dirs:
# #     print(model_dir)
#     log = TOOLS.load_log(model_dir)
#     hp = TOOLS.load_hp(model_dir)
#     try:
#         analysis = clustering.Analysis(model_dir, 'rule')
#         # Plots from instance methods in class
#         # analysis.plot_cluster_score()
#         analysis.plot_variance()
#         # analysis.plot_similarity_matrix()
#         analysis.plot_2Dvisualization()
#         # analysis.plot_example_unit()
#         # analysis.plot_connectivity_byclusters()
#
#         log['n_cluster'] = analysis.n_cluster
#         log['model_dir'] = model_dir
#         TOOLS.save_log(log)
#     except IOError:
#         # Training never finished
#         assert log['perf_min'][-1] <= hp['target_perf']
#
#     # analysis.plot_example_unit()
#     # analysis.plot_variance()
#     # analysis.plot_2Dvisualization()
#
#     print("done")
#
# compute_n_cluster(model_dir)
#
#
# import platform
#
# # Get the bitness of the Python interpreter
# python_bitness = platform.architecture()[0]