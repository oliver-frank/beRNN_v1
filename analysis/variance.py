########################################################################################################################
# info: Variance
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division

import os
# import time
import numpy as np
import pickle
from collections import OrderedDict
# import matplotlib as mpl
# import matplotlib.pyplot as plt
import tensorflow as tf
import random
import errno

# from task import *
from network import Model
from training import createSplittedDatasets, create_cMask
import tools

save = True

########################################################################################################################
# Define functions
########################################################################################################################
def _compute_variance_bymodel(data_dir, model_dir, layer, data_type, networkAnalysis, model: object, sess: object, mode: str, monthsConsidered: list, rules: object = None, random_rotation: object = False) -> object:
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

    # Fallback for some scenarios
    hp.setdefault('rng', np.random.default_rng())
    # model_dir = model.model_dir

    if rules is None:
        rules = hp['rules']
    print(rules)

    if hp.get('multiLayer') == True:
        # n_hidden = hp['n_rnn_per_layer'][layer-1]
        n_hidden = 0
        for n_rnn_layer in hp['n_rnn_per_layer']: # info: Accumulate number of hidden units
            n_hidden += n_rnn_layer
    else:
        n_hidden = hp['n_rnn']

    if random_rotation:
        # Generate random orthogonal matrix
        from scipy.stats import ortho_group
        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    # data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data\\BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '\\PreprocessedData_wResp_ALL'
    month = '_'.join(model_dir.split('_')[-2:]) # only current model's month considered
    train_data, eval_data = createSplittedDatasets(hp, data_dir, month)

    # Skip taskVariance creation if already exists
    save_name = 'var_' + mode + '_lay' + str(layer) + '_' + data_type
    if random_rotation:
        save_name += '_rr'
    fname = os.path.join(model_dir, save_name + '.pkl')

    if os.path.exists(fname) == False:
        try:

            for task in rules:
                # print(task)
                if mode == 'train':
                    data = train_data
                elif mode == 'test':
                    data = eval_data

                x, y, y_loc, response = tools.load_trials(hp['rng'], task, mode, hp['batch_size'], data, False)
                epochs = tools.find_epochs(x)

                c_mask = create_cMask(y, response, hp, mode)

                # # info: ################################################################################################
                # fixation_steps = tools.getEpochSteps(y)
                #
                # # Creat c_mask for current batch
                # if hp['loss_type'] == 'lsq':
                #     c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
                #     for i in range(y.shape[1]):
                #         # Fixation epoch
                #         c_mask[:fixation_steps, i, :] = 1.
                #         # Response epoch
                #         c_mask[fixation_steps:, i, :] = 1.
                #
                #     # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
                #     c_mask[:, :, 0] *= 2.  # Fixation is important
                #     c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))
                #     c_mask /= c_mask.mean()
                #
                # else:
                #     c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
                #     for i in range(y.shape[1]):
                #         # Fixation epoch
                #         c_mask[:fixation_steps, i, :] = 1.
                #         # Response epoch
                #         c_mask[fixation_steps:, i, :] = 1.
                #
                #     c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
                #     c_mask /= c_mask.mean()
                # # info: ################################################################################################

                feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)

                if hp.get('multiLayer') == True: # info: Only concatenate layers for hpOverview & taskRepresentation
                    h = sess.run(model.h_all_layers, feed_dict=feed_dict)
                    h = np.concatenate([h[0], h[1], h[2]], axis=-1)
                elif hp.get('multiLayer') == True and hp.get('networkAnalysis') == True: # info: Visualization of performance and funcCorrelation for all layers individually
                    h = sess.run(model.h_all_layers, feed_dict=feed_dict)
                    h = h[layer-1] # info: choose the layer of current interest
                elif hp.get('multiLayer') == False or hp.get('multiLayer') == None:
                    h = sess.run(model.h, feed_dict=feed_dict)

                if random_rotation:
                    h = np.dot(h, random_ortho_matrix)  # randomly rotate

                for e_name, e_time in epochs.items():
                    if 'fix' not in e_name:  # Ignore fixation period
                        h_all_byepoch[(task, e_name)] = h[e_time[0]:e_time[1], :,:]

                # Ignore fixation period
                h_all_byrule[task] = h[epochs['fix1'][1]:, :, :]

                # Reorder h_all_byepoch by epoch-first
                keys = list(h_all_byepoch.keys())
                # ind_key_sort = np.lexsort(zip(*keys))
                # Using mergesort because it is stable
                ind_key_sort = np.argsort(list(zip(*keys))[1], kind='mergesort')
                h_all_byepoch = OrderedDict([(keys[i], h_all_byepoch[keys[i]]) for i in ind_key_sort])

                if data_type == 'rule':
                    h_all = h_all_byrule # info: only task orientated over all epochs (fixation excluded)
                elif data_type == 'epoch':
                    h_all = h_all_byepoch # info: task and epoch orientated (fixation excluded), similar to h_all_byrule as there is only fixation and response epoch

                h_var_all = np.zeros((n_hidden, len(h_all.keys())))
                for i, val in enumerate(h_all.values()):
                    # info: Iterating through all tasks and creating an individual task variance value representing distribution
                    #  of average unit activities over trials in current batch (n_rnn_units x variance value for average unit
                    #  activity over trials)
                    # val is Time, Batch, Units
                    # Variance across time and stimulus
                    # h_var_all[:, i] = val[t_start:].reshape((-1, n_hidden)).var(axis=0)
                    # Variance acros stimulus, then averaged across time
                    h_var_all[:, i] = val.var(axis=1).mean(axis=0)

                result = {'h_var_all': h_var_all, 'keys': list(h_all.keys())}
                save_name = 'var_' + mode + '_lay' + str(layer) + '_' + data_type

                if random_rotation:
                    save_name += '_rr'

                fname = os.path.join(model_dir, save_name + '.pkl')
                # dir_path = os.path.dirname(fname)

                # C:\\Users\\oliver.frank\\Desktop\\PyProjects\\beRNNmodels\\2025_04_multiLayerHpGridSearch\\1\\beRNN_03_AllTask_3-5_data_highDim_correctOnly_3stimTC_iteration0_LeakyRNN_64-64-64_relu-relu-linear\\model_month_3\\variance_test_layer0_rule_data_highDim_correctOnly_3stimTC.pkl
                # C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\2025_04_multiLayerHpGridSearch\1\beRNN_03_AllTask_3-5_data_highDim_correctOnly_3stimTC_iteration0_LeakyRNN_64-64-64_relu-relu-linear\model_month_3

        except Exception as e:
            print(f"Error in _compute_variance_bymodel: {e}")

        try:
            os.makedirs(os.path.dirname(fname), exist_ok=True)

            print("Saving to:", fname)
            print("Dir to create:", os.path.dirname(fname))
            print("Exists:", os.path.exists(os.path.dirname(fname)))

            print("model_dir =", model_dir)

            with open(fname, 'wb') as f:
                pickle.dump(result, f)
            print(f"Variance file saved: {fname}")

        except OSError as e:
            if e.errno == errno.EEXIST:
                print(f"Directory already exists: {fname}")
            else:
                print(f"OS Error while saving file: {e}")
        except Exception as e:
            print(f"Failed to save variance file {fname}: {e}")

    else:
        print(f"task variance file: {fname} already exists. Skipping repetition.")
    return fname

def _compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, networkAnalysis, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    model = Model(model_dir, sigma_rec=0)
    with tf.Session() as sess:
        model.restore()
        fname = _compute_variance_bymodel(data_dir, model_dir, layer, data_type, networkAnalysis, model, sess, mode, monthsConsidered, rules, random_rotation)
        return fname

def compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, networkAnalysis, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    dirs = tools.valid_model_dirs(model_dir)
    for d in dirs:
        fname = _compute_variance(data_dir, d, layer, mode, monthsConsidered, data_type, networkAnalysis, rules, random_rotation)
        return fname


if __name__ == '__main__':
    pass


