
########################################################################################################################
# todo: LAB ############################################################################################################
########################################################################################################################


# Error Comparison #####################################################################################################
from __future__ import division
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
import glob
import os

import Tools
from Network import Model

model = 'Model_145_BeRNN_05_Month_2-6'
model_dir = 'W://group_csp//analyses//oliver.frank//BeRNN_models' + '//' + model + '//'
trial_dir = 'W://group_csp//analyses//oliver.frank//Data' + '//BeRNN_' + model.split('_')[3] + '//PreprocessedData_wResp_ALL'
monthsConsidered = list(range(int(model.split('_')[-1].split('-')[0]),int(model.split('_')[-1].split('-')[1])+1))

# ?
Tools.mkdir_p(model_dir)
# Load hp
hp = Tools.load_hp(model_dir)
# Load log
log = Tools.load_log(model_dir)
# Build the model
model = Model(model_dir, hp=hp) # Do I really need that?

with tf.Session() as sess:
    model.restore(model_dir)  # complete restore

    hp = model.hp
    mode = 'Evaluation'

    #############################################################################
    log = do_eval(sess, model, log, trial_dir, hp['rule_trains'])
    #############################################################################

    for rule_test in hp['rules']:
        n_rep = 125 # 8 trials on each batch - So 1000 trials for each task
        batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        # clsq_tmp = list()
        # creg_tmp = list()
        # perf_tmp = list()
        for i_rep in range(n_rep):
            x,y,y_loc,file_stem = Tools.load_trials(trial_dir, monthsConsidered, rule_test, mode)

            # todo: ################################################################################################
            fixation_steps = Tools.getEpochSteps(y,file_stem)

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
                c_mask = c_mask.reshape((y.shape[0]*y.shape[1], y.shape[2]))

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

            feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp) # y: participnt response, that gives the lable for what the network is trained for
            # print('passed feed_dict Evaluation')
            # print(feed_dict)
            # print('x',type(x),x.shape)
            # print('y',type(y),y.shape)
            # print('y_loc',type(y_loc),y_loc.shape)
            c_lsq, c_reg, y_hat_test = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],feed_dict=feed_dict)


participantDirectory = trial_dir + '//RP'
npy_files = glob.glob(os.path.join(participantDirectory, '*Response.npy'))
# Convert all elements in monthsConsidered to strings
# monthsConsidered = [str(month) for month in monthsConsidered]
# # Select the files only for a certain task
# selected_months_files = [file for file in npy_files if any(month in file for month in monthsConsidered)]

Response = np.load(npy_files[3], allow_pickle=True)



# Open log for time
import json

# Define the path to the JSON file
json_file_path = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models\\Model_141_BeRNN_05_Month_2-4\\log.json'

# Open and read the JSON file
with open(json_file_path, 'r') as file:
    data = json.load(file)
