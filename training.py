########################################################################################################################
# info: training
########################################################################################################################
# Train Models with collected data. Particpant, data and directories have to be adjusted manually.
########################################################################################################################

########################################################################################################################
# Import necessary libraries and modules
########################################################################################################################
from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import time
from collections import defaultdict

import os
import numpy as np
# import matplotlib.pyplot as plt
import tensorflow as tf
import random

from network import Model, get_perf,get_perf_lowDIM
# from analysis import variance
import tools


########################################################################################################################
# Predefine functions
########################################################################################################################
def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''

    num_ring = tools.get_num_ring(ruleset)
    n_rule = tools.get_num_rule(ruleset)

    machine = 'local' # 'local' 'pandora' 'hitkip'
    data = 'data_highDim' # 'data_highDim' , data_highDim_correctOnly , data_highDim_lowCognition , data_lowDim , data_lowDim_correctOnly , data_lowDim_lowCognition, 'data_highDim_correctOnly_3stimTC'
    trainingBatch = '02_errorUnbalanced'
    trainingYear_Month = 'errorBalancingTest'

    if 'highDim' in data: # fix: lowDim_timeCompressed needs to be skipped here
        n_eachring = 32
        n_outputring = n_eachring
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1
    else:
        n_eachring = 10
        n_outputring = 2
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_outputring + 1

    hp = {
        # batch size for training and evaluations
        'batch_size': 80, # 20/40/80/120/160
        # 'batch_size_test': 640, # batch_size for testing
        'in_type': 'normal', # input type: normal, multi
        'rnn_type': 'LeakyRNN', # Type of RNNs: NonRecurrent, LeakyRNN, LeakyGRU, EILeakyGRU | GRU, LSTM
        'multiLayer': False, # only applicaple with LeakyRNN
        'n_rnn': 128,  # number of recurrent units for one hidden layer architecture
        'activation': 'relu',  # Type of activation runctions, relu, softplus, tanh, elu, linear
        'n_rnn_per_layer': [256, 128, 64],
        'activations_per_layer': ['relu', 'tanh', 'linear'],
        'loss_type': 'lsq', # # Type of loss functions - Cross-entropy loss
        'optimizer': 'adam', # 'adam', 'sgd'
        'tau': 50, # # Time constant (ms)- default 100
        'dt': 20, # discretization time step (ms) .
        # 'alpha': 0.2, # (redundant) discretization time step/time constant - dt/tau = alpha - ratio decides on how much previous states are taken into account for current state - low alpha more memory, high alpha more forgetting - alpha * h(t-1)
        'sigma_rec': 0.01, # recurrent noise - directly influencing the noise added to the network
        'sigma_x': 0.01, # input noise
        'w_rec_init': 'randortho', # leaky_rec weight initialization, diag, randortho, randgauss, brainStructure (only accessible with LeakyRNN : 32-256)
        'l1_h': 1e-4, # l1 lambda (regularizing with absolute value of magnitude of coefficients, leading to sparse features)
        'l2_h': 5e-6, # l2 lambda (regularizing with squared value of magnitude of coefficients, decreasing influence of features)
        'l1_weight': 1e-5, # l2 regularization on weight
        'l2_weight': 1e-4, # l2 regularization on weight
        'l2_weight_init': 0, # l2 regularization on deviation from initialization
        'p_weight_train': None, # proportion of weights not to be regularized, None or float between (0, 1) - 1-p_weight_train will be multiplied by w_mask_value
        'w_mask_value': 0.1, # default .1 - value that will be multiplied with L2 regularization (combined with p_weight_train), <1 will decrease it
        'target_perf': 1.0, # Stopping performance
        'n_eachring': n_eachring, # number of units each ring
        'num_ring': num_ring, # number of rings
        'n_rule': n_rule, # number of rules
        'rule_start': 1 + num_ring * n_eachring, # first input index for rule units
        'n_input': n_input, # number of input units
        'n_output': n_output, # number of output units
        'rng': np.random.default_rng(), # np.random.RandomState(seed=0), random number used for several random initializations
        'ruleset': ruleset, # number of input units
        'save_name': 'test', # name to save
        'learning_rate': 0.0015, # learning rate
        'learning_rate_mode': None, # Will overwrite learning_rate if it is not None - 'triangular', 'triangular2', 'exp_range', 'decay'
        'base_lr': [1e-5],
        'max_lr': [1e-3],
        'errorBalancingValue': 1., # will be multiplied with c_mask_responseValue for objective error trials - 1. means no difference between errors and corrects are made
        'c_mask_responseValue': 5., # c_mask response epoch value - strenght response epoch is taken into account for error calculation
        's_mask': None, # 'sc1000', None - info: only accesible on local machine
        'rule_probs': None, # Rule probabilities to be drawn
        'use_separate_input': False,  # whether rule and stimulus inputs are represented separately
        # 'c_intsyn': 0, # intelligent synapses parameters, tuple (c, ksi) -> Yang et al. only apply these in sequential training
        # 'ksi_intsyn': 0,
        'monthsConsidered': ['month_3', 'month_4', 'month_5'], # months to train and test
        'monthsString': '3-5', # monthsTaken
        # 'rule_prob_map': {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1},
        'rule_prob_map': {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}, # fraction of tasks represented in training data
        'tasksString': 'Alltask', # tasks taken
        'sequenceMode': True, # Decide if models are trained sequentially month-wise
        'participant': 'beRNN_05', # Participant to take
        'data': data, # 'data_highDim' , data_highDim_correctOnly , data_highDim_lowCognition , data_lowDim , data_lowDim_correctOnly , data_lowDim_lowCognition, data_timeCompressed, data_lowDim_timeCompressed
        'machine': machine,
        'trainingBatch': trainingBatch,
        'trainingYear_Month': trainingYear_Month
    }

    return hp

def split_files(files, split_ratio=0.8):
    # random.seed(42) # attention: add seed to always shuffle similiar - would be good for NetworkAnalysis as it iwll result in robust solutions
    random.seed(np.random.default_rng()) # attention: add seed to always shuffle similiar - would be good for NetworkAnalysis as it iwll result in robust solutions
    random.shuffle(files)
    split_index = int(len(files) * split_ratio)
    return files[:split_index], files[split_index:]

def do_eval(sess, model, log, rule_train, eval_data):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    mode = 'test'
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for task in hp['rules']:
        n_rep = 20 # 20 * 40 or 20 * 20 trials per evaluation are taken, depending on batch_size
        # batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()


        for i_rep in range(n_rep):
            try:
                x,y,y_loc,response = tools.load_trials(task, mode, hp['batch_size'], eval_data, False)  # y_loc is participantResponse_perfEvalForm

                # info: ################################################################################################
                fixation_steps = tools.getEpochSteps(y)
                if fixation_steps == None:  # if no fixation_steps could be found
                    continue

                # Creat c_mask for current batch
                if hp['loss_type'] == 'lsq':
                    c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')
                    for i in range(y.shape[1]):
                        # Fixation epoch
                        c_mask[:fixation_steps, i, :] = 1.
                        # Response epoch
                        c_mask[fixation_steps:, i, :] = hp['c_mask_responseValue'] # info: 1 or 5

                    # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
                    # c_mask[:, :, 0] *= 2.  # Fixation is important
                    c_mask = c_mask.reshape((y.shape[0]*y.shape[1], y.shape[2]))

                else:
                    c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
                    for i in range(y.shape[1]):
                        # Fixation epoch
                        c_mask[:fixation_steps, i, :] = 1.
                        # Response epoch
                        c_mask[fixation_steps:, i, :] = hp['c_mask_responseValue'] # info: 1 or 5

                    c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
                    c_mask /= c_mask.mean()

                # info: ################################################################################################

                feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp) # y: participnt response, that gives the lable for what the network is trained for
                # print('passed feed_dict Evaluation')
                # print(feed_dict)
                # print('x',type(x),x.shape)
                # print('y',type(y),y.shape)
                # print('y_loc',type(y_loc),y_loc.shape)
                c_lsq, c_reg, y_hat_test = sess.run([model.cost_lsq, model.cost_reg, model.y_hat],feed_dict=feed_dict)
                # print('passed sess.run')
                # Cost is first summed over time,
                # and averaged across batch and units
                # We did the averaging over time through c_mask

                if 'lowDim' in hp['data']:
                    perf_test = np.mean(get_perf_lowDIM(y_hat_test, y_loc))
                else:
                    perf_test = np.mean(get_perf(y_hat_test, y_loc)) # info: y_loc is participant response as groundTruth
                print('perf_test   ', perf_test)
                clsq_tmp.append(c_lsq)
                creg_tmp.append(c_reg)
                perf_tmp.append(perf_test)

            except Exception as e:
                print(f"Error during evaluation of task {task}, iteration {i_rep}: {e}")
                continue  # Skip iteration on error

        # If no valid results were collected, continue with next task
        if not clsq_tmp or not perf_tmp:
            print(f"Skipping task {task} as no valid data was processed.")
            continue

        log['cost_' + task].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_' + task].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_' + task].append(np.mean(perf_tmp, dtype=np.float64))

        print('{:15s}'.format(task) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # info: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]

    try:
        perf_tests_mean = np.mean([log['perf_' + r][-1] for r in rule_tmp])
        log['perf_avg'].append(perf_tests_mean)
        perf_tests_min = np.min([log['perf_' + r][-1] for r in rule_tmp])
        log['perf_min'].append(perf_tests_min)
    except KeyError as e:
        print(f"Warning: Could not compute final performance due to missing key {e}.")
        return log

    # Save the model and log
    try:
        model.save()
        tools.save_log(log)
    except Exception as e:
        print(f"Warning: Could not save model/log due to {e}.")

    return log

def train(model_dir,train_data ,eval_data,hp=None,max_steps=3e6,display_step=500,ruleset='all',rule_trains=None,rule_prob_map=None,seed=0,
          load_dir=None,trainables=None):
    """Train the network.

    Args:
        model_dir: str, training directory
        hp: dictionary of hyperparameters
        max_steps: int, maximum number of training steps
        display_step: int, display steps
        ruleset: the set of rules to train
        rule_trains: list of rules to train, if None then all rules possible
        rule_prob_map: None or dictionary of relative rule probability
        seed: int, random seed to be used

    Returns:
        model is stored at model_dir/model.ckpt
        training configuration is stored at model_dir/hp.json
    """

    tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    # default_hp = get_default_hp('all')
    if hp is not None:
        default_hp.update(hp) # fix: Where does this update function come from?
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.default_rng()

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = tools.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if hp['rule_prob_map'] is None:
        hp['rule_prob_map'] = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array([hp['rule_prob_map'].get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))
    tools.save_hp(hp, model_dir)

    # # info: Create structural mask to multiply with hidden layer
    # if hp['s_mask'] == 'sc1000':
    #     import scipy.io
    #     sc = scipy.io.loadmat('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\art_beRNN\\masks\\sc1000')
    #     # sc = scipy.io.loadmat('C:\\Users\\oliver.frank\\Desktop\\PyProjects\\art_beRNN\\masks\\sc100')
    #     sc_mask = sc['mat_zero'] # 1000
    #     # sc_mask = sc['shaefer_rsn'] # 100
    #
    #     # info: quadratic mask matrix necessary - attention: maskSize = numberHiddenUnits !
    #     maskSize = sc_mask.shape[0]
    #     for i in range(0, maskSize):
    #         for j in range(0, maskSize):
    #             sc_mask[i, j] = 1 if sc_mask[i, j] > 11 else 0

        # import numpy as np
        # count_ones = np.count_nonzero(sc_mask[0,:] == 1) # info: 495 hidden units are trained if threshold = ß
        #
        # # info: Visualize the structural matrix
        # import matplotlib.pyplot as plt
        #
        # plt.figure(figsize=(8, 8))
        # plt.imshow(sc_mask, aspect='auto', cmap='coolwarm')
        # plt.colorbar()
        # plt.title("Visualization of a 1000x1000 ndarray")
        # plt.show()

        # hp['s_mask'] = sc_mask
    # elif # fix: Add other structural masks here


    # Build the model
    model = Model(model_dir, hp=hp)

    # Display hp
    for key, val in hp.items():
        print('{:20s} = '.format(key) + str(val))

    # Store results
    log = defaultdict(list)
    log['model_dir'] = model_dir

    # Record time
    t_start = time.time()
    # Count loaded trials/batches
    trialsLoaded = 0

    with tf.Session() as sess:
        if load_dir is not None:
            model.restore(load_dir)  # complete restore
            print('model restored')
        else:
            # Assume everything is restored
            sess.run(tf.global_variables_initializer())

        # Set trainable parameters
        if trainables is None or trainables == 'all':
            var_list = model.var_list  # train everything
        elif trainables == 'input':
            # train all nputs
            var_list = [v for v in model.var_list if ('input' in v.name) and ('rnn' not in v.name)]
        elif trainables == 'rule':
            # train rule inputs only
            var_list = [v for v in model.var_list if 'rule_input' in v.name]
        else:
            raise ValueError('Unknown trainables')
        model.set_optimizer(var_list=var_list)

        # penalty on deviation from initial weight
        if hp['l2_weight_init'] > 0:
            anchor_ws = sess.run(model.weight_list)
            for w, w_val in zip(model.weight_list, anchor_ws):
                model.cost_reg += (hp['l2_weight_init'] * tf.nn.l2_loss(w - w_val))

            model.set_optimizer(var_list=var_list)

        # partial weight training
        # Explanation: In summary, this code introduces a form of partial weight training by applying L2 regularization
        # only to a subset of the weights. The subset is determined by random masking, controlled by the hyperparameter
        # 'p_weight_train'. All weights below the p_weight_train threshold won't be trained in this iteration.
        if ('p_weight_train' in hp and
                (hp['p_weight_train'] is not None) and
                hp['p_weight_train'] < 1.0):
            for w in model.weight_list:
                w_val = sess.run(w)
                w_size = sess.run(tf.size(w))
                w_mask_tmp = np.linspace(0, 1, w_size)
                hp['rng'].shuffle(w_mask_tmp)
                ind_fix = w_mask_tmp > hp['p_weight_train']
                w_mask = np.zeros(w_size, dtype=np.float32)
                w_mask[ind_fix] = hp['w_mask_value'] # 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer(var_list=var_list)

        step = 0
        while step * hp['batch_size'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0: # III: Every 500 steps (20000 trials) do the evaluation
                    log['trials'].append(step * hp['batch_size'])
                    log['times'].append(time.time() - t_start)
                    log = do_eval(sess, model, log, hp['rule_trains'],eval_data)
                    elapsed_time = time.time() - t_start  # Calculate elapsed time
                    print(f"Elapsed time after batch number {trialsLoaded}: {elapsed_time:.2f} seconds")
                    # After training
                    total_time = time.time() - t_start
                    print(f"Total training time: {total_time:.2f} seconds")
                    # if log['perf_avg'][-1] > model.hp['target_perf']:
                    # check if minimum performance is above target
                    if log['perf_min'][-1] > model.hp['target_perf']:
                        print('Perf reached the target: {:0.2f}'.format(
                            hp['target_perf']))
                        break

                    # if rich_output:
                    #     display_rich_output(model, sess, step, log, model_dir)

                # Training
                task = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
                # Generate a random batch of trials; each batch has the same trial length
                mode = 'train'
                x,y,y_loc,response = tools.load_trials(task,mode,hp['batch_size'], train_data, False) # y_loc is participantResponse_perfEvalForm

                # info: ################################################################################################
                fixation_steps = tools.getEpochSteps(y)
                if fixation_steps == None: # if no fixation_steps could be found
                    continue
                # Creat c_mask for current batch
                if hp['loss_type'] == 'lsq':
                    c_mask = np.zeros((y.shape[0], y.shape[1], y.shape[2]), dtype='float32')

                    # fix: Create a c_mask that emphasizes errors by multiplying the error contribution for backProp by 5. and corrects by 1.
                    errorBalancingVector = np.zeros(response.shape[1], dtype='float32')
                    for j in range(response.shape[1]):
                        if response[0][j] == response[1][j]:
                            errorBalancingVector[j] = 1. # weight value for corrects
                        else:
                            errorBalancingVector[j] = hp['errorBalancingValue']

                    for i in range(y.shape[1]):
                        # Fixation epoch
                        c_mask[:fixation_steps, i, :] = 1.
                        # Response epoch
                        c_mask[fixation_steps:, i, :] = hp['c_mask_responseValue'] * errorBalancingVector[i] # info: 1 or 5

                    # self.c_mask[:, :, 0] *= self.n_eachring # Fixation is important
                    # c_mask[:, :, 0] *= 2.  # Fixation is important # info: with or without
                    c_mask = c_mask.reshape((y.shape[0]*y.shape[1], y.shape[2]))
                    c_mask /= c_mask.mean()

                else:
                    c_mask = np.zeros((y.shape[0], y.shape[1]), dtype='float32')
                    for i in range(y.shape[1]):
                        # Fixation epoch
                        c_mask[:fixation_steps, i, :] = 1.
                        # Response epoch
                        c_mask[fixation_steps:, i, :] = hp['c_mask_responseValue'] # info: 1 or 5

                    c_mask = c_mask.reshape((y.shape[0] * y.shape[1],))
                    c_mask /= c_mask.mean()

                # info: ################################################################################################

                trialsLoaded += 1

                # Generating feed_dict.
                feed_dict = tools.gen_feed_dict(model, x, y, c_mask, hp)
                # print('passed feed_dict Training')
                # print(feed_dict)

                sess.run(model.train_step, feed_dict=feed_dict) # info: Trainables are actualized - train_step should represent the step in training.py and the global_step in network.py

                # Get Training performance in a similiar fashion as in do_eval
                clsq_train_tmp = list()
                creg_train_tmp = list()
                perf_train_tmp = list()
                c_lsq_train, c_reg_train, y_hat_train = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict) # info: lsq+reg = total_loss - updates the network parameters

                if 'lowDim' in hp['data']:
                    perf_train = np.round(np.mean(get_perf_lowDIM(y_hat_train, y_loc)),3)  # info: y_loc is participant response as groundTruth
                else:
                    perf_train = np.round(np.mean(get_perf(y_hat_train, y_loc)),3) # info: y_loc is participant response as groundTruth
                print('perf_train   ', perf_train)
                clsq_train_tmp.append(c_lsq_train)
                creg_train_tmp.append(c_reg_train)
                perf_train_tmp.append(perf_train)

                log['cost_train_' + task].append(np.mean(clsq_train_tmp, dtype=np.float64))
                log['creg_train_' + task].append(np.mean(creg_train_tmp, dtype=np.float64))
                log['perf_train_' + task].append(np.mean(perf_train_tmp, dtype=np.float64))

                print('{:15s}'.format(task) +
                      '| train cost {:0.6f}'.format(np.mean(clsq_train_tmp)) +
                      '| train c_reg {:0.6f}'.format(np.mean(c_reg_train)) +
                      '  | train perf {:0.2f}'.format(np.mean(perf_train)))

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")


########################################################################################################################
# Train model
########################################################################################################################
if __name__ == '__main__':
    # Initialize list for all training times for each model
    trainingTimeList = []
    for modelNumber in range(1,2): # Define number of iterations and models to be created for every month, respectively

        # Measure time for every model, respectively
        trainingTimeTotal_hours = 0
        # Start it
        start_time = time.perf_counter()
        print(f'START TRAINING MODEL: {modelNumber}')

        hp = get_default_hp('all')
        load_dir = None

        # Define main path
        if hp['machine'] == 'local':
            path = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects'
        elif hp['machine'] == 'hitkip':
            path = '/zi/home/oliver.frank/Desktop'
        elif hp['machine'] == 'pandora':
            path = '/pandora/home/oliver.frank/01_Projects/RNN/multitask_BeRNN-main'

        # Define data path
        preprocessedData_path = os.path.join(path, 'Data', hp['participant'], hp['data'])  # pandora

        for month in hp['monthsConsidered']: # attention: You have to delete this if cascade training should be set OFF
            # Adjust variables manually as needed
            model_name = f'model_{month}'


            # Define model_dir for different servers
            if hp['machine'] == 'local':
                if hp['multiLayer'] == True:
                    numberOfLayers = len(hp['n_rnn_per_layer'])
                    if numberOfLayers == 2:
                        model_dir = os.path.join(
                            f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['trainingBatch']}\\{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}",model_name)
                    else:
                        model_dir = os.path.join(f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['trainingBatch']}\\{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}-{hp['n_rnn_per_layer'][2]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}-{hp['activations_per_layer'][2]}", model_name)
                else:
                    model_dir = os.path.join(f"{path}\\beRNNmodels\\{hp['trainingYear_Month']}\\{hp['trainingBatch']}\\{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn']}_{hp['activation']}",model_name)

            elif hp['machine'] == 'hitkip' or hp['machine'] == 'pandora':
                if hp['multiLayer'] == True:
                    numberOfLayers = len(hp['n_rnn_per_layer'])
                    if numberOfLayers == 2:
                        model_dir = os.path.join(
                            f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['trainingBatch']}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}",model_name)
                    else:
                        model_dir = os.path.join(
                            f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['trainingBatch']}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn_per_layer'][0]}-{hp['n_rnn_per_layer'][1]}-{hp['n_rnn_per_layer'][2]}_{hp['activations_per_layer'][0]}-{hp['activations_per_layer'][1]}-{hp['activations_per_layer'][2]}",model_name)
                else:
                    model_dir = os.path.join(
                        f"{path}/beRNNmodels/{hp['trainingYear_Month']}/{hp['trainingBatch']}/{hp['participant']}_{hp['tasksString']}_{hp['monthsString']}_{hp['data']}_iteration{modelNumber}_{hp['rnn_type']}_{hp['w_rec_init']}_{hp['n_rnn']}_{hp['activation']}",model_name)


            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Split the data into training and test data -----------------------------------------------------------------------
            # List of the subdirectories
            subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]

            # Initialize dictionaries to store training and evaluation data
            train_data = {}
            eval_data = {}

            for subdir in subdirs:
                # Collect all file triplets in the current subdirectory
                file_quartett = []
                for file in os.listdir(subdir):
                    if file.endswith('Input.npy'):
                        # # III: Exclude files with specific substrings in their names
                        # if any(exclude in file for exclude in ['Randomization', 'Segmentation', 'Mirrored', 'Rotation']):
                        #     continue
                        # Include only files that contain any of the months in monthsConsidered
                        if month not in file: # Sort out months which should not be considered
                            continue
                        # Add all necessary files to triplets
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

            # info: If you want to initialize the new model with an old one
            # load_dir = 'C:\\Users\\oliver.frank\\Desktop\\PyProjects\\beRNNmodels\\2025_03\\sc_mask_final\\beRNN_03_All_3-5_data_highDim_correctOnly_iteration1_LeakyRNN_1000_relu\\model_month_3'
            # Start Training ---------------------------------------------------------------------------------------------------
            train(model_dir=model_dir, train_data = train_data, eval_data = eval_data, load_dir = load_dir)

            # info: If True previous model parameters will be taken to initialize consecutive model, creating sequential training
            if hp['sequenceMode'] == True:
                load_dir = model_dir

        end_time = time.perf_counter()
        # Training time taken into account
        elapsed_time_seconds = end_time - start_time
        elapsed_time_minutes = elapsed_time_seconds / 60
        elapsed_time_hours = elapsed_time_minutes / 60

        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_seconds:.2f} seconds")
        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_minutes:.2f} minutes")
        print(f"TIME TAKEN TO TRAIN MODEL {modelNumber}: {elapsed_time_hours:.2f} hours")

        # Accumulate trainingTime
        trainingTimeList.append(elapsed_time_hours)
        trainingTimeTotal_hours += elapsed_time_hours

    # Save training time total and list to folder as a text file
    file_path = os.path.join(model_dir, 'times.txt')

    with open(file_path, 'w') as f:
        f.write(f"training time total (hours): {trainingTimeTotal_hours}\n")
        f.write("training time each individual model (hours):\n")
        for time in trainingTimeList:
            f.write(f"{time}\n")

    print(f"Training times saved to {file_path}")


