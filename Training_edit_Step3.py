"""Main training loop"""
#%%
from __future__ import division

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import time
from collections import defaultdict

import os
import numpy as np
# import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random

from Network import Model, get_perf
# from analysis import variance
import Tools


#%%

def get_default_hp(ruleset):
    '''Get a default hp.

    Useful for debugging.

    Returns:
        hp : a dictionary containing training hpuration
    '''
    num_ring = Tools.get_num_ring(ruleset)
    n_rule = Tools.get_num_rule(ruleset)

    n_eachring = 32
    n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
    hp = {
        # batch size for training and evaluation
        'batch_size': 40,
        # batch_size for testing
        # 'batch_size_test': 640,
        # input type: normal, multi
        'in_type': 'normal',
        # Type of RNNs: NonRecurrent, LeakyRNN, LeakyGRU, EILeakyGRU, GRU, LSTM
        'rnn_type': 'LeakyRNN',
        # whether rule and stimulus inputs are represented separately
        'use_separate_input': False,
        # Type of loss functions
        'loss_type': 'lsq',
        # Optimizer
        'optimizer': 'adam',
        # Type of activation runctions, relu, softplus, tanh, elu, linear
        'activation': 'softplus',
        # Time constant (ms)
        'tau': 100,
        # discretization time step (ms)
        'dt': 20,
        # discretization time step/time constant
        'alpha': 0.2,
        # recurrent noise - directly influencing the noise added to the network; can prevent over-fitting especially when learning time sequences
        'sigma_rec': 0.05,
        # input noise
        'sigma_x': 0.01,
        # leaky_rec weight initialization, diag, randortho, randgauss
        'w_rec_init': 'randortho',
        # a default weak regularization prevents instability (regularizing with absolute value of magnitude of coefficients, leading to sparse features)
        'l1_h': 0,
        # l2 regularization on activity (regularizing with squared value of magnitude of coefficients, decreasing influence of features)
        'l2_h': 0.00001,
        # l2 regularization on weight
        'l1_weight': 0,
        # l2 regularization on weight
        'l2_weight': 0,
        # l2 regularization on deviation from initialization
        'l2_weight_init': 0,
        # proportion of weights to train, None or float between (0, 1) - e.g. .1 will train a random 10% weight selection, the rest stays fixed (Yang et al. range: .05-.075)
        'p_weight_train': None,
        # Stopping performance
        'target_perf': 1.0,
        # number of units each ring
        'n_eachring': n_eachring,
        # number of rings
        'num_ring': num_ring,
        # number of rules
        'n_rule': n_rule,
        # first input index for rule units
        'rule_start': 1 + num_ring * n_eachring,
        # number of input units
        'n_input': n_input,
        # number of output units
        'n_output': n_output,
        # number of recurrent units
        'n_rnn': 256,
        # random number used for several random initializations
        'rng': np.random.RandomState(seed=0),
        # number of input units
        'ruleset': ruleset,
        # name to save
        'save_name': 'test',
        # learning rate
        'learning_rate': 0.001,
        # intelligent synapses parameters, tuple (c, ksi) -> Yang et al. only apply these in sequential training
        # 'c_intsyn': 0,
        # 'ksi_intsyn': 0,
    }

    return hp

def do_eval(sess, model, log, trial_dir, rule_train, monthsConsidered, eval_data):
    """Do evaluation.

    Args:
        sess: tensorflow session
        model: Model class instance
        log: dictionary that stores the log
        rule_train: string or list of strings, the rules being trained
    """
    hp = model.hp
    mode = 'Evaluation'
    if not hasattr(rule_train, '__iter__'):
        rule_name_print = rule_train
    else:
        rule_name_print = ' & '.join(rule_train)

    print('Trial {:7d}'.format(log['trials'][-1]) +
          '  | Time {:0.2f} s'.format(log['times'][-1]) +
          '  | Now training ' + rule_name_print)

    for rule_test in hp['rules']:
        n_rep = 20 # 20 * 40 or 20 * 20 trials per evaluation are taken, depending on batch_size
        # batch_size_test_rep = int(hp['batch_size_test']/n_rep)
        clsq_tmp = list()
        creg_tmp = list()
        perf_tmp = list()
        for i_rep in range(n_rep):
            x,y,y_loc = Tools.load_trials(trial_dir, monthsConsidered, rule_test, mode, hp['batch_size'], eval_data)  # y_loc is participantResponse_perfEvalForm

            # todo: ################################################################################################
            fixation_steps = Tools.getEpochSteps(y)

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
            # print('passed sess.run')
            # Cost is first summed over time,
            # and averaged across batch and units
            # We did the averaging over time through c_mask
            perf_test = np.mean(get_perf(y_hat_test, y_loc)) # todo: y_loc is participant response as groundTruth
            clsq_tmp.append(c_lsq)
            creg_tmp.append(c_reg)
            perf_tmp.append(perf_test)

        log['cost_' + rule_test].append(np.mean(clsq_tmp, dtype=np.float64))
        log['creg_' + rule_test].append(np.mean(creg_tmp, dtype=np.float64))
        log['perf_' + rule_test].append(np.mean(perf_tmp, dtype=np.float64))
        print('{:15s}'.format(rule_test) +
              '| cost {:0.6f}'.format(np.mean(clsq_tmp)) +
              '| c_reg {:0.6f}'.format(np.mean(creg_tmp)) +
              '  | perf {:0.2f}'.format(np.mean(perf_tmp)))
        sys.stdout.flush()

    # TODO: This needs to be fixed since now rules are strings
    if hasattr(rule_train, '__iter__'):
        rule_tmp = rule_train
    else:
        rule_tmp = [rule_train]
    perf_tests_mean = np.mean([log['perf_' + r][-1] for r in rule_tmp])
    log['perf_avg'].append(perf_tests_mean)

    perf_tests_min = np.min([log['perf_' + r][-1] for r in rule_tmp])
    log['perf_min'].append(perf_tests_min)

    # Saving the model
    model.save()
    Tools.save_log(log)

    return log

def train(model_dir,trial_dir,monthsConsidered,train_data ,eval_data,hp=None,max_steps=3e6,display_step=1000,ruleset='all',rule_trains=None,rule_prob_map=None,seed=0,
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

    Tools.mkdir_p(model_dir)

    # Network parameters
    default_hp = get_default_hp(ruleset)
    # default_hp = get_default_hp('all')
    if hp is not None:
        default_hp.update(hp)
    hp = default_hp
    hp['seed'] = seed
    hp['rng'] = np.random.RandomState(seed)

    # Rules to train and test. Rules in a set are trained together
    if rule_trains is None:
        # By default, training all rules available to this ruleset
        hp['rule_trains'] = Tools.rules_dict[ruleset]
    else:
        hp['rule_trains'] = rule_trains
    hp['rules'] = hp['rule_trains']

    # Assign probabilities for rule_trains.
    if rule_prob_map is None:
        rule_prob_map = dict()

    # Turn into rule_trains format
    hp['rule_probs'] = None
    if hasattr(hp['rule_trains'], '__iter__'):
        # Set default as 1.
        rule_prob = np.array([rule_prob_map.get(r, 1.) for r in hp['rule_trains']])
        hp['rule_probs'] = list(rule_prob / np.sum(rule_prob))
    Tools.save_hp(hp, model_dir)

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
                w_mask[ind_fix] = 1e-1  # will be squared in l2_loss
                w_mask = tf.constant(w_mask)
                w_mask = tf.reshape(w_mask, w.shape)
                model.cost_reg += tf.nn.l2_loss((w - w_val) * w_mask)
            model.set_optimizer(var_list=var_list)

        step = 0
        while step * hp['batch_size'] <= max_steps:
            try:
                # Validation
                if step % display_step == 0:
                    log['trials'].append(step * hp['batch_size'])
                    log['times'].append(time.time() - t_start)
                    log = do_eval(sess, model, log, trial_dir, hp['rule_trains'],monthsConsidered, eval_data)
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
                rule_train_now = hp['rng'].choice(hp['rule_trains'], p=hp['rule_probs'])
                # Generate a random batch of trials; each batch has the same trial length
                mode = 'Training'
                x,y,y_loc = Tools.load_trials(trial_dir,monthsConsidered,rule_train_now,mode,hp['batch_size'], train_data) # y_loc is participantResponse_perfEvalForm

                # todo: ################################################################################################
                fixation_steps = Tools.getEpochSteps(y)
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

                trialsLoaded += 1

                # Generating feed_dict.
                feed_dict = Tools.gen_feed_dict(model, x, y, c_mask, hp)
                # print('passed feed_dict Training')
                # print(feed_dict)
                sess.run(model.train_step, feed_dict=feed_dict)

                # Get Training performance in a similiar fashion as in do_eval
                clsq_train_tmp = list()
                creg_train_tmp = list()
                perf_train_tmp = list()
                c_lsq_train, c_reg_train, y_hat_train = sess.run([model.cost_lsq, model.cost_reg, model.y_hat], feed_dict=feed_dict)
                perf_train = np.mean(get_perf(y_hat_train, y_loc)) # todo: y_loc is participant response as groundTruth
                clsq_train_tmp.append(c_lsq_train)
                creg_train_tmp.append(c_reg_train)
                perf_train_tmp.append(perf_train)

                log['cost_train_' + rule_train_now].append(np.mean(clsq_train_tmp, dtype=np.float64))
                log['creg_train_' + rule_train_now].append(np.mean(creg_train_tmp, dtype=np.float64))
                log['perf_train_' + rule_train_now].append(np.mean(perf_train_tmp, dtype=np.float64))

                print('{:15s}'.format(rule_train_now) +
                      '| train cost {:0.6f}'.format(np.mean(clsq_train_tmp)) +
                      '| train c_reg {:0.6f}'.format(np.mean(c_reg_train)) +
                      '  | train perf {:0.2f}'.format(np.mean(perf_train)))

                step += 1

            except KeyboardInterrupt:
                print("Optimization interrupted by user")
                break

        print("Optimization finished!")


#%%
# # ERRORS_REMOVED
# if __name__ == '__main__':
#     dataFolder = "Data"
#     participant = 'BeRNN_01'
#     model_folder = 'Model'
#     strToSave = '2-6'
#     number = str(150)
#     model_number = 'XModel_' + number + '_' + participant + '_Month_' + strToSave # Manually add months considered e.g. 1-7
#     monthsConsidered = ['2','3','4','5','6','7'] # Add all months you want to take into consideration for training and evaluation
#     model_dir = os.path.join('/Users/marcschubert/Documents/rnns/models', model_number)
#     model_dir = os.path.join(model_dir, "ERRORS_REMOVED")

#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)
#     # preprocessedData_folder = 'PreprocessedData_wResp_ALL'
#     # preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', dataFolder, participant, 'PreprocessedData_wResp_ALL')
#     preprocessedData_path = os.path.join('/Users/marcschubert/Documents/rnns/Data/', participant,'PreprocessedData_wResp_ALL')
#     # Define probability of each task being trained
#     rule_prob_map = {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}
#     rule_prob_map = {"DM": 1,"DM_Anti": 1}#,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}
#     # III: Split the data ##############################################################################################
#     # List of the subdirectories
#     subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]


#     # Initialize dictionaries to store training and evaluation data
#     train_data = {}
#     eval_data = {}

#     # Function to split the files
#     def split_files(files, split_ratio=0.8):
#         random.shuffle(files)
#         split_index = int(len(files) * split_ratio)
#         return files[:split_index], files[split_index:]
    
#     print(subdirs)
#     for subdir in subdirs:
#         # Collect all file triplets in the current subdirectory
#         file_triplets = []
#         for file in os.listdir(subdir):
#             if file.find('Input')>0:
#                 base_name = file.split('Input')[0]

#                 if base_name.find("WM")> 0:
#                     print(base_name)
#                     error_group = ""  # "ORIGNAL"
#                 else: 
#                     error_group = "_ERRORS_REMOVED"  # "ORIGNAL"


#                 #error_group = "_ERRORS_ONLY"  # "ORIGNAL"
#                 input_file = os.path.join(subdir, base_name + 'Input'+error_group+'.npy')
#                 yloc_file = os.path.join(subdir, base_name + 'yLoc'+error_group+'.npy')
#                 output_file = os.path.join(subdir, base_name + 'Output'+error_group+'.npy')
#                 file_triplets.append((input_file, yloc_file, output_file))


#         # Split the file triplets
#         train_files, eval_files = split_files(file_triplets)

#         # Store the results in the dictionaries
#         train_data[subdir] = train_files
#         eval_data[subdir] = eval_files
#     # III: Split the data ##############################################################################################

#     train(model_dir=model_dir, trial_dir=preprocessedData_path, monthsConsidered = monthsConsidered, rule_prob_map=rule_prob_map, train_data = train_data, eval_data = eval_data)
#     # train(model_dir=model_dir, trial_dir=preprocessedData_path)



#%%
# ORIGINAL 

# getSubsetsAndSave(cor_only, "coronly")
#     getSubsetsAndSave(cor_sys, "corsys")
#     getSubsetsAndSave(cor_rand, "corrand")
#     getSubsetsAndSave(sys_only, "sysonly")
#     getSubsetsAndSave(rand_only, "randonly")
#     getSubsetsAndSave(sysrand, "sysrand")
#     getSubsetsAndSave(org, "org")

run_ids = ["org", "coronly", "corsys", "corrand", "sysrand",  "randonly","sysonly"]

if __name__ == '__main__':

    for CUR_RUN_ID in run_ids:
        print(CUR_RUN_ID)
        dataFolder = "Data"
        participant = 'BeRNN_01'
        model_folder = 'Model'
        strToSave = '2-6'
        number = str(150)
        model_number = 'XModel_' + number + '_' + participant + '_Month_' + strToSave # Manually add months considered e.g. 1-7
        monthsConsidered = ['2','3','4','5','6','7'] # Add all months you want to take into consideration for training and evaluation
        model_dir = os.path.join('/Users/marcschubert/Documents/rnns/models', model_number)
        model_dir = os.path.join(model_dir, "ORIGINAL_" + CUR_RUN_ID)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # preprocessedData_folder = 'PreprocessedData_wResp_ALL'
        # preprocessedData_path = os.path.join('W:\\group_csp\\analyses\\oliver.frank', dataFolder, participant, 'PreprocessedData_wResp_ALL')
        preprocessedData_path = os.path.join('/Users/marcschubert/Documents/rnns/Data/', participant,'PreprocessedData_wResp_ALL')
        # Define probability of each task being trained
        rule_prob_map = {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}
        rule_prob_map = {"DM": 1,"DM_Anti": 1,"EF": 1,"EF_Anti": 1,"RP": 1,"RP_Anti": 1,"RP_Ctx1": 1,"RP_Ctx2": 1,"WM": 1,"WM_Anti": 1,"WM_Ctx1": 1,"WM_Ctx2": 1}
        # III: Split the data ##############################################################################################
        # List of the subdirectories
        subdirs = [os.path.join(preprocessedData_path, d) for d in os.listdir(preprocessedData_path) if os.path.isdir(os.path.join(preprocessedData_path, d))]


        # Initialize dictionaries to store training and evaluation data
        train_data = {}
        eval_data = {}

        # Function to split the files
        def split_files(files, split_ratio=0.8):
            random.shuffle(files)
            split_index = int(len(files) * split_ratio)
            return files[:split_index], files[split_index:]
        
        print(subdirs)
        for subdir in subdirs:
            # Collect all file triplets in the current subdirectory
            file_triplets = []
            for file in os.listdir(subdir):
                if file.find('Input')>0:
                    base_name = file.split('Input')[0]

                    if base_name.find("WM")> 0:
                        print(base_name)
                        error_group = ""  # "ORIGNAL"
                    else: 
                        error_group = "_ORIGINAL" + "_"+CUR_RUN_ID  # "ORIGNAL"


                    #error_group = "_ERRORS_ONLY"  # "ORIGNAL"
                    input_file = os.path.join(subdir, base_name + 'Input'+error_group+'.npy')
                    yloc_file = os.path.join(subdir, base_name + 'yLoc'+error_group+'.npy')
                    output_file = os.path.join(subdir, base_name + 'Output'+error_group+'.npy')
                    file_triplets.append((input_file, yloc_file, output_file))


            # Split the file triplets
            train_files, eval_files = split_files(file_triplets)

            # Store the results in the dictionaries
            train_data[subdir] = train_files
            eval_data[subdir] = eval_files
        # III: Split the data ##############################################################################################

        train(model_dir=model_dir, trial_dir=preprocessedData_path, monthsConsidered = monthsConsidered, rule_prob_map=rule_prob_map, train_data = train_data, eval_data = eval_data)
        # train(model_dir=model_dir, trial_dir=preprocessedData_path)

  

#%%#####
