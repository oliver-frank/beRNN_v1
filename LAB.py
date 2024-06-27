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
from Network import Model, popvec, tf_popvec

model = 'Model_145_BeRNN_05_Month_2-6'
model_dir = 'W:\\group_csp\\analyses\\oliver.frank\\BeRNN_models' + '\\' + model + '\\'
trial_dir = 'W:\\group_csp\\analyses\\oliver.frank\\Data' + '\\BeRNN_' + model.split('_')[3] + '\\PreprocessedData_wResp_ALL'
monthsConsidered = list(range(int(model.split('_')[-1].split('-')[0]),int(model.split('_')[-1].split('-')[1])+1))
task = 'RP'

npy_files_Input = glob.glob(os.path.join(trial_dir, task, '*Input.npy'))

# Initialize count variables Match and Mismatch
errorMatch = 0
errorMisMatch = 0
correctMatch = 0
correctMisMatch = 0
noCount = 0

hp = Tools.load_hp(model_dir)
model = Model(model_dir, hp=hp)

# Get networkResponse by opening model and let it process on the current trial
with tf.Session() as sess:
    # if load_dir is not None:
    #     model.restore(load_dir)  # complete restore
    # else:
    # Assume everything is restored
    sess.run(tf.global_variables_initializer())
    # # Load model
    # saver = tf.train.Saver()
    # saver.restore(sess, tf.train.latest_checkpoint(model_dir))

    for file in npy_files_Input:
        # Get responses for one arbitrary trial particpantResponse, groundTruth, networkResponse
        trialInput = np.load(npy_files_Input[0], allow_pickle=True)
        participantResponse = np.load(os.path.join(trial_dir, task, '-'.join(npy_files_Input[0].split('\\')[-1].split('-')[:-1]) + '-Output.npy'),allow_pickle=True)
        participantResponse_perfEvalForm = np.load(os.path.join(trial_dir, task, '-'.join(npy_files_Input[0].split('\\')[-1].split('-')[:-1]) + '-yLoc.npy'),allow_pickle=True)
        groundTruth = np.load(os.path.join(trial_dir, task, '-'.join(npy_files_Input[0].split('\\')[-1].split('-')[0:5]) + '-Response.npy'),allow_pickle=True)
        # todo: sort groundTruth

        # c_mask actually not needed, as we won't train the network here
        c_mask = np.zeros((participantResponse.shape[0]*participantResponse.shape[1], participantResponse.shape[2]),dtype='float32')
        # Get model response for current batch
        feed_dict = Tools.gen_feed_dict(model, trialInput, participantResponse, c_mask, hp)  # y: participnt response, that gives the lable for what the network is trained for
        c_lsq, c_reg, y_hat_test = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)
        # Save model response for current batch
        modelResponse_machineForm = y_hat_test
        print('One modelResponse created')
        popVec_modelResponse_machineForm = popvec(modelResponse_machineForm[..., 1:])
        print(popVec_modelResponse_machineForm)

        # todo: Get the value representing the response direction of the model for one trail (in the complete batch)
        # todo: and get the field representing it in the human form to find in the InputTrial to which response it took
        def get_perf(y_hat, y_loc):
            """Get performance.

            Args:
              y_hat: Actual output. Numpy array (Time, Batch, Unit)
              y_loc: Target output location (-1 for fixation).
                Numpy array (Time, Batch)

            Returns:
              perf: Numpy array (Batch,)
            """
            if len(y_hat.shape) != 3:
                raise ValueError('y_hat must have shape (Time, Batch, Unit)')
            # Only look at last time points
            y_loc = y_loc[-1]
            y_hat = y_hat[-1]

            # Fixation and location of y_hat
            y_hat_fix = y_hat[..., 0]
            y_hat_loc = popvec(y_hat[..., 1:])

            # Fixating? Correctly saccading?
            fixating = y_hat_fix > 0.5

            original_dist = y_loc - y_hat_loc
            dist = np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))
            corr_loc = dist < 0.2 * np.pi  # 35 degreee margin around exact correct respond

            # Should fixate?
            should_fix = y_loc < 0

            # performance
            perf = should_fix * fixating + (1 - should_fix) * corr_loc * (1 - fixating)
            return perf


















        # todo: Categorize modelResponse into one discrete response option

        # ##############################################################################################################
        # # todo: Compare the different responses - propably somehow over the evaluation function
        # responseClass_participantResponse = compare(participantResponse, groundTruth)
        # responseClass_networkResponse = compare(networkResponse, groundTruth)
        #
        # todo: Append all the files to a dataframe
        # # Count Match/Mismatch Table
        # if responseClass_participantResponse[0] == 'Error' and responseClass_networkResponse[0] == 'Error' and responseClass_participantResponse[1] \
        #     == networkResponse[1]:
        #     errorMatch += 1
        # elif responseClass_participantResponse[0] == 'Error' and responseClass_networkResponse[0] == 'Error' and responseClass_participantResponse[1] \
        #     != networkResponse[1]:
        #     errorMisMatch += 1
        # elif responseClass_participantResponse[0] == 'Correct' and responseClass_networkResponse[0] == 'Correct' and responseClass_participantResponse[1] \
        #     == networkResponse[1]:
        #     correctMatch += 1
        # elif responseClass_participantResponse[0] == 'Correct' and responseClass_networkResponse[0] == 'Correct' and responseClass_participantResponse[1] \
        #     != networkResponse[1]:
        #     correctMisMatch += 1
        # else:
        #     noCount += 1
