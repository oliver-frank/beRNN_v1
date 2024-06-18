"""
Compute Variance
"""

from __future__ import division

import os
# import time
import numpy as np
import pickle
from collections import OrderedDict
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import tensorflow as tf

# from task import *
from Network import Model
import Tools

save = True


def _compute_variance_bymodel(model: object, sess: object, mode: str, monthsConsidered: list, rules: object = None, random_rotation: object = False) -> object:
    """Compute variance for all tasks.

        Args:
            model: network.Model instance
            sess: tensorflow session
            rules: list of rules to compute variance, list of strings
            random_rotation: boolean. If True, rotate the neural activity.
        """
    h_all_byrule = OrderedDict()
    h_all_byepoch = OrderedDict()
    hp = model.hp
    model_dir = model.model_dir


    if rules is None:
        rules = hp['rules']
    print(rules)

    n_hidden = hp['n_rnn']

    if random_rotation:
        # Generate random orthogonal matrix
        from scipy.stats import ortho_group
        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    for rule in rules:
        print(rule)
        trial_dir = 'W:\\group_csp\\analyses\\oliver.frank' + '\\Data\\BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '\\PreprocessedData_wResp_ALL'
        x, y, y_loc, file_stem = Tools.load_trials(trial_dir, monthsConsidered, rule, mode)
        epochs = Tools.find_epochs(x)

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
        h = sess.run(model.h, feed_dict=feed_dict)

        if random_rotation:
            h = np.dot(h, random_ortho_matrix)  # randomly rotate

        for e_name, e_time in epochs.items():
            if 'fix' not in e_name:  # Ignore fixation period
                h_all_byepoch[(rule, e_name)] = h[e_time[0]:e_time[1], :,:]

        # Ignore fixation period
        h_all_byrule[rule] = h[epochs['fix1'][1]:, :, :]

    # Reorder h_all_byepoch by epoch-first
    keys = list(h_all_byepoch.keys())
    # ind_key_sort = np.lexsort(zip(*keys))
    # Using mergesort because it is stable
    ind_key_sort = np.argsort(list(zip(*keys))[1], kind='mergesort')
    h_all_byepoch = OrderedDict(
        [(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

    for data_type in ['rule', 'epoch']:
        if data_type == 'rule':
            h_all = h_all_byrule
        elif data_type == 'epoch':
            h_all = h_all_byepoch
        else:
            raise ValueError

        h_var_all = np.zeros((n_hidden, len(h_all.keys())))
        for i, val in enumerate(h_all.values()):
            # val is Time, Batch, Units
            # Variance across time and stimulus
            # h_var_all[:, i] = val[t_start:].reshape((-1, n_hidden)).var(axis=0)
            # Variance acros stimulus, then averaged across time
            h_var_all[:, i] = val.var(axis=1).mean(axis=0)

        result = {'h_var_all': h_var_all, 'keys': list(h_all.keys())}
        save_name = 'variance_' + mode + '_' + data_type
        if random_rotation:
            save_name += '_rr'

        fname = os.path.join(model.model_dir, save_name + '.pkl')
        print('Variance saved at {:s}'.format(fname))
        with open(fname, 'wb') as f:
            pickle.dump(result, f)

def _compute_variance(model_dir, mode, monthsConsidered, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    model = Model(model_dir, sigma_rec=0)
    with tf.Session() as sess:
        model.restore()
        _compute_variance_bymodel(model, sess, mode, monthsConsidered, rules, random_rotation)

def compute_variance(model_dir, mode, monthsConsidered, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    dirs = Tools.valid_model_dirs(model_dir)
    for d in dirs:
        _compute_variance(d, mode, monthsConsidered, rules, random_rotation)


if __name__ == '__main__':
    pass

