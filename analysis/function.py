from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import Tools
from Network import Model



########################################################################################################################
def easy_activity_plot(model_dir, rule, monthsConsidered, mode):
    """A simple plot of neural activity from one task.

    Args:
        model_dir: directory where model file is saved
        rule: string, the rule to plot
    """

    model = Model(model_dir)
    hp = model.hp

    with tf.Session() as sess:
        model.restore()

        trial_dir = 'W://group_csp//analyses//oliver.frank//Data//BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '//PreprocessedData_wResp_ALL'

        x, y, y_loc, file_stem = Tools.load_trials(trial_dir, monthsConsidered, rule, mode)

        # todo: ################################################################################################
        fixation_steps = Tools.getEpochSteps(y, file_stem)

        # Creat c_mask for current batch
        if hp['loss_type'] == 'lsq':
            c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important
            c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        else:
            c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
            c_mask /= c_mask.mean()

        # todo: ################################################################################################

        feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

        # trial = generate_trials(rule, hp, mode='test')
        # feed_dict = Tools.gen_feed_dict(model, trial, hp)
        # h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)
        # # All matrices have shape (n_time, n_condition, n_neuron)

    # Take only the one example trial
    i_trial = 0

    for activity, title in zip([x, h, y_hat],['input', 'recurrent', 'output']):
        plt.figure()
        plt.imshow(activity[:,i_trial,:].T, aspect='auto', cmap='hot', interpolation='none', origin='lower')
        plt.title(title)
        plt.colorbar()
        plt.show()

model_dir = 'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_112_BeRNN_01_Month_2-4'
rule = 'DM'
monthsConsidered = ['2','3','4']
mode = 'Evaluation'

easy_activity_plot(model_dir, rule, monthsConsidered, mode)
########################################################################################################################



########################################################################################################################
def pretty_inputoutput_plot(model_dir, rule, monthsConsidered, save=False, plot_ylabel=False):
    """Plot the input and output activity for a sample trial from one task.

    Args:
        model_dir: model directory
        rule: string, the rule
        save: bool, whether to save plots
        plot_ylabel: bool, whether to plot ylable
    """


    fs = 7

    model = Model(model_dir)
    hp = model.hp
    trial_dir = 'W://group_csp//analyses//oliver.frank//Data//BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '//PreprocessedData_wResp_ALL'

    with tf.Session() as sess:
        model.restore()

        x, y, y_loc, file_stem = Tools.load_trials(trial_dir, monthsConsidered, rule, mode)

        # todo: ################################################################################################
        fixation_steps = Tools.getEpochSteps(y, file_stem)

        # Creat c_mask for current batch
        if hp['loss_type'] == 'lsq':
            c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important
            c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))

        else:
            c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 5.

            c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
            c_mask /= c_mask.mean()

        # todo: ################################################################################################

        feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp)
        h, y_hat = sess.run([model.h, model.y_hat], feed_dict=feed_dict)

        t_plot = np.arange(x.shape[0])*hp['dt']/1000

        assert hp['num_ring'] == 2

        n_eachring = hp['n_eachring']

        fig = plt.figure(figsize=(1.3,2))
        ylabels = ['fix. in', 'stim. mod1', 'stim. mod2','fix. out', 'out']
        heights = np.array([0.03,0.2,0.2,0.03,0.2])+0.01
        for i in range(5):
            ax = fig.add_axes([0.15,sum(heights[i+1:]+0.02)+0.1,0.8,heights[i]])
            cmap = 'Purples'
            plt.xticks([])
            ax.tick_params(axis='both', which='major', labelsize=fs,
                           width=0.5, length=2, pad=3)

            if plot_ylabel:
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('bottom')
                ax.yaxis.set_ticks_position('left')

            else:
                ax.spines["left"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.xaxis.set_ticks_position('none')

            if i == 0:
                plt.plot(t_plot, x[:,0,0], color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0,1],['',''],rotation='vertical')
                plt.ylim([-0.1,1.5])
                plt.title(rule,fontsize=fs)
            elif i == 1:
                plt.imshow(x[:,0,1:1+n_eachring].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none',origin='lower')
                if plot_ylabel:
                    plt.yticks([0, (n_eachring-1)/2, n_eachring-1],
                               [r'0$\degree$',r'180$\degree$',r'360$\degree$'],
                               rotation='vertical')
            elif i == 2:
                plt.imshow(x[:, 0, 1+n_eachring:1+2*n_eachring].T,
                           aspect='auto', cmap=cmap, vmin=0, vmax=1,
                           interpolation='none',origin='lower')

                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
            elif i == 3:
                plt.plot(t_plot, y[:,0,0],color='xkcd:green')
                plt.plot(t_plot, y_hat[:,0,0],color='xkcd:blue')
                if plot_ylabel:
                    plt.yticks([0.05,0.8],['',''],rotation='vertical')
                plt.ylim([-0.1,1.1])
            elif i == 4:
                plt.imshow(y_hat[:, 0, 1:].T, aspect='auto', cmap=cmap,
                           vmin=0, vmax=1, interpolation='none', origin='lower')
                if plot_ylabel:
                    plt.yticks(
                        [0, (n_eachring-1)/2, n_eachring-1],
                        [r'0$\degree$', r'180$\degree$', r'360$\degree$'],
                        rotation='vertical')
                plt.xticks([0,y_hat.shape[0]], ['0', '2'])
                plt.xlabel('Time (s)',fontsize=fs, labelpad=-3)
                ax.spines["bottom"].set_visible(True)

            if plot_ylabel:
               plt.ylabel(ylabels[i],fontsize=fs)
            else:
                plt.yticks([])
            ax.get_yaxis().set_label_coords(-0.12,0.5)

        if save:
            save_name = 'figure/sample_'+ rule +'.pdf'
            plt.savefig(save_name, transparent=True)
        plt.show()

        # plt.figure()
        # _ = plt.plot(h_sample[:,0,:20])
        # plt.show()
        #
        # plt.figure()
        # _ = plt.plot(y_sample[:,0,:])
        # plt.show()

model_dir = 'W:\\group_csp\\analyses\\oliver.frank' + '\BeRNN_models\Model_112_BeRNN_01_Month_2-4'
rule = 'DM'
monthsConsidered = ['2','3','4']
mode = 'Evaluation'

pretty_inputoutput_plot(model_dir, rule, monthsConsidered, save=False, plot_ylabel=False)
########################################################################################################################



########################################################################################################################
def plot_connectivity_byclusters(model_dir):
    """Plot connectivity of the model"""

    # Sort data by labels and by input connectivity
    model = Model(model_dir)
    hp = model.hp
    ind_active = model.ind_active

    with tf.Session() as sess:
        model.restore()
        w_in = sess.run(model.w_in).T
        w_rec = sess.run(model.w_rec).T
        w_out = sess.run(model.w_out).T
        b_rec = sess.run(model.b_rec)
        b_out = sess.run(model.b_out)

    w_rec = w_rec[ind_active, :][:, ind_active]
    w_in = w_in[ind_active, :]
    w_out = w_out[:, ind_active]
    b_rec = b_rec[ind_active]

    # nx, nh, ny = hp['shape']
    nr = hp['n_eachring']

    sort_by = 'w_in'
    if sort_by == 'w_in':
        w_in_mod1 = w_in[:, 1:nr + 1]
        w_in_mod2 = w_in[:, nr + 1:2 * nr + 1]
        w_in_modboth = w_in_mod1 + w_in_mod2
        w_prefs = np.argmax(w_in_modboth, axis=1)
    elif sort_by == 'w_out':
        w_prefs = np.argmax(w_out[1:], axis=0)

    # sort by labels then by prefs
    ind_sort = np.lexsort((w_prefs, self.labels))

    ######################### Plotting Connectivity ###############################
    nx = self.hp['n_input']
    ny = self.hp['n_output']
    nh = len(self.ind_active)
    nr = self.hp['n_eachring']
    nrule = len(self.hp['rules'])

    # Plot active units
    _w_rec = w_rec[ind_sort, :][:, ind_sort]
    _w_in = w_in[ind_sort, :]
    _w_out = w_out[:, ind_sort]
    _b_rec = b_rec[ind_sort, np.newaxis]
    _b_out = b_out[:, np.newaxis]
    labels = self.labels[ind_sort]

    l = 0.3
    l0 = (1 - 1.5 * l) / nh

    plot_infos = [(_w_rec, [l, l, nh * l0, nh * l0]),
                  (_w_in[:, [0]], [l - (nx + 15) * l0, l, 1 * l0, nh * l0]),  # Fixation input
                  (_w_in[:, 1:nr + 1], [l - (nx + 11) * l0, l, nr * l0, nh * l0]),  # Mod 1 stimulus
                  (_w_in[:, nr + 1:2 * nr + 1], [l - (nx - nr + 8) * l0, l, nr * l0, nh * l0]),  # Mod 2 stimulus
                  (_w_in[:, 2 * nr + 1:], [l - (nx - 2 * nr + 5) * l0, l, nrule * l0, nh * l0]),  # Rule inputs
                  (_w_out[[0], :], [l, l - (4) * l0, nh * l0, 1 * l0]),
                  (_w_out[1:, :], [l, l - (ny + 6) * l0, nh * l0, (ny - 1) * l0]),
                  (_b_rec, [l + (nh + 6) * l0, l, l0, nh * l0]),
                  (_b_out, [l + (nh + 6) * l0, l - (ny + 6) * l0, l0, ny * l0])]

    # cmap = sns.diverging_palette(220, 10, sep=80, as_cmap=True)
    cmap = 'coolwarm'
    fig = plt.figure(figsize=(6, 6))
    for plot_info in plot_infos:
        ax = fig.add_axes(plot_info[1])
        vmin, vmid, vmax = np.percentile(plot_info[0].flatten(), [5, 50, 95])
        _ = ax.imshow(plot_info[0], interpolation='nearest', cmap=cmap, aspect='auto',
                      vmin=vmid - (vmax - vmin) / 2, vmax=vmid + (vmax - vmin) / 2)
        ax.axis('off')

    ax1 = fig.add_axes([l, l + nh * l0, nh * l0, 6 * l0])
    ax2 = fig.add_axes([l - 6 * l0, l, 6 * l0, nh * l0])
    for il, l in enumerate(self.unique_labels):
        ind_l = np.where(labels == l)[0][[0, -1]] + np.array([0, 1])
        ax1.plot(ind_l, [0, 0], linewidth=2, solid_capstyle='butt',
                 color=kelly_colors[il + 1])
        ax2.plot([0, 0], len(labels) - ind_l, linewidth=2, solid_capstyle='butt',
                 color=kelly_colors[il + 1])
    ax1.set_xlim([0, len(labels)])
    ax2.set_ylim([0, len(labels)])
    ax1.axis('off')
    ax2.axis('off')
    if save:
        plt.savefig('figure/connectivity_by' + self.data_type + '.pdf', transparent=True)
    plt.show()