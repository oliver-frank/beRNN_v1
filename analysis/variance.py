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
import tools

save = True

########################################################################################################################
# Define functions
########################################################################################################################
def _compute_variance_bymodel(data_dir, model_dir, layer, data_type, model: object, sess: object, mode: str, monthsConsidered: list, rules: object = None, random_rotation: object = False) -> object:
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
    # model_dir = model.model_dir

    if rules is None:
        rules = hp['rules']
    print(rules)

    # # Do a task variance analysis for each hidden layer
    # if hp.get('multiLayer') == True:
    #     numberOfLayers = len(hp['n_rnn_per_layer'])
    # else:
    #     numberOfLayers = 1

    # for layer in range(0, numberOfLayers):
    if hp.get('multiLayer') == True:
        n_hidden = hp['n_rnn_per_layer'][layer-1]
    else:
        n_hidden = hp['n_rnn']

    if random_rotation:
        # Generate random orthogonal matrix
        from scipy.stats import ortho_group
        random_ortho_matrix = ortho_group.rvs(dim=n_hidden)

    # data_dir = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data\\BeRNN_' + model_dir.split('BeRNN_')[-1].split('_')[0] + '\\PreprocessedData_wResp_ALL'

    # III: Split the data ##############################################################################################
    # List of the subdirectories
    subdirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    # Initialize dictionaries to store training and evaluation data
    train_data = {}
    eval_data = {}

    # Function to split the files
    def split_files(files, split_ratio=0.8):
        random.seed(42) # info: add seed to always shuffle similiar
        random.shuffle(files)
        split_index = int(len(files) * split_ratio)
        return files[:split_index], files[split_index:]

    for subdir in subdirs:
        # Collect all file triplets in the current subdirectory
        file_quartett = []
        for file in os.listdir(subdir):
            if file.endswith('Input.npy'):
                # # III: Exclude files with specific substrings in their names
                # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                #     continue
                # Include only files that contain any of the months in monthsConsidered
                if not any(month in file for month in monthsConsidered):
                    continue
                base_name = file.split('Input')[0]
                input_file = os.path.join(subdir, base_name + 'Input.npy')
                yloc_file = os.path.join(subdir, base_name + 'yLoc.npy')
                output_file = os.path.join(subdir, base_name + 'Output.npy')
                response_file = os.path.join(subdir, base_name + 'Response.npy')

                file_quartett.append((input_file, yloc_file, output_file, response_file))

        # Split the file triplets
        train_files, eval_files = split_files(file_quartett)

        # Store the results in the dictionaries
        train_data[subdir] = train_files
        eval_data[subdir] = eval_files
    # III: Split the data ##############################################################################################
    for task in rules:
        # print(task)
        if mode == 'train':
            data = train_data
        elif mode == 'test':
            data = eval_data
        x, y, y_loc, response = tools.load_trials(task, mode, hp['batch_size'], data, False)
        epochs = tools.find_epochs(x)

        # info: ################################################################################################
        fixation_steps = tools.getEpochSteps(y)

        # Creat c_mask for current batch
        if hp['loss_type'] == 'lsq':
            c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 1.

            # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
            c_mask[:, :, 0] *= 2.  # Fixation is important
            c_mask = c_mask.reshape((y.shape[0] * y.shape[1], y.shape[2]))
            c_mask /= c_mask.mean()

        else:
            c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
            for i in range(y.shape[1]):
                # Fixation epoch
                c_mask[:fixation_steps, i, :] = 1.
                # Response epoch
                c_mask[fixation_steps:, i, :] = 1.

            c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
            c_mask /= c_mask.mean()
        # info: ################################################################################################

        feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)

        if hp.get('multiLayer') == True:
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

    return fname

def _compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    model = Model(model_dir, sigma_rec=0)
    with tf.Session() as sess:
        model.restore()
        fname = _compute_variance_bymodel(data_dir, model_dir, layer, data_type, model, sess, mode, monthsConsidered, rules, random_rotation)
        return fname

def compute_variance(data_dir, model_dir, layer, mode, monthsConsidered, data_type, rules=None, random_rotation=False):
    """Compute variance for all tasks.

    Args:
        model_dir: str, the path of the model directory
        rules: list of rules to compute variance, list of strings
        random_rotation: boolean. If True, rotate the neural activity.
    """
    dirs = tools.valid_model_dirs(model_dir)
    for d in dirs:
        fname = _compute_variance(data_dir, d, layer, mode, monthsConsidered, data_type, rules, random_rotation)
        return fname


if __name__ == '__main__':
    pass


