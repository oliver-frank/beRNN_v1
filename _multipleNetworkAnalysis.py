import os
import re
import json
import numpy as np
import networkx as nx
import seaborn as sns
import pandas as pd
import itertools

from pathlib import Path
import scipy.stats
from scipy.stats import ttest_rel
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import cosine
# from scipy.spatial.distance import pdist, squareform

from matplotlib import pyplot as plt
import matplotlib.patches as patches

from networkx.algorithms.community import greedy_modularity_communities, modularity

from _training import apply_density_threshold
from _analysis import clustering
from tools import load_hp, load_pickle, participation_coefficient

'''
Complete brain/beRNN comparison by correlation and/or rsa. 
Both data modalities have to be preprocessed with:

- beRNN: beRNN_v1/_hyperparameterOverview.py and beRNN_v1/taskRepresentation.py
- brain: 'preprocess_fRMI2rdm' below
'''

########################################################################################################################
# head - Variables and functions #######################################################################################
########################################################################################################################
setup = {
    'comparison': ['correlation', 'rsa', None][0],
    'modalityWithin_comparison': ['brain', 'beRNN', 'standard'][2], # standard is beRNN brain comparison
    'numberOfModels': [20, 20], # second value represents beRNN_04 - only defined for beRNNs - should be 3 if compared to brain in 'standard' - [5, 3] or [20, 20]
    'threshold': 1.0,
    'participants_beRNN': ['beRNN_03', 'beRNN_04', 'beRNN_01', 'beRNN_02', 'beRNN_05'], # order for paper - only applied for RDA
    'paper_nomenclatur': ['HC1', 'HC2', 'MDD', 'ASD', 'SCZ'], # nomenclatur for paper plots - only applied for RDA
    'participants': ['sub-KPB84', 'sub-YL4AS', 'sub-6IECX', 'sub-DKHPB', 'sub-96WID'],
    'participants_snip': ['sub-SNIPKPB84', 'sub-SNIPYL4AS', 'sub-SNIP6IECX', 'sub-SNIPDKHPB', 'sub-SNIP96WID'],
    'folder_beRNN': fr'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\_robustnessTest_multiTask_beRNN_01_highDimCorrects_256_hp_2',
    'folder_brain': r'W:\group_csp\analyses\oliver.frank\share',
    'folder_topologicalMarker_pValue_lists': r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists',
    # 'folder_brain_meanVecs': r'W:\group_csp\in_house_datasets\bernn\mri\derivatives\xcpd_0.11.1\sub-SNIP6IECX01\func',
    'dataType': None, # will be defined below
    # 'rdm_taskset': 'representationalDissimilarity_cosine_fundamentals',
    'rsa_directory': r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__rsaVisuals',
    'directory_rdm': r'W:\group_csp\analyses\oliver.frank\share\functional_matrices_rdm',
    'tasks': ['reward', 'flanker', 'faces', 'nback'],
    'preprocess_fMRI2rdm': False,
    "visualize_fMRI": False,
    "visualize_dti": False,
    "correlationOf_correlationMatices_fMRI": False,
    "correlationOf_correlationMatices_beRNN": False,
    "plotTopMarkerOverDensities": True,
    "session": '03'
}

if 'highDim_correctOnly' in setup['folder_beRNN'] or 'highDimCorrects' in setup['folder_beRNN']:
    setup['dataType'] = 'highDim_correctOnly'
elif 'highDim' in setup['folder_beRNN']:
    setup['dataType'] = 'highDim'

beRNN_brain_dict = {
    'sub-KPB84': 'beRNN_01',
    'sub-YL4AS': 'beRNN_02',
    'sub-6IECX': 'beRNN_03',
    'sub-DKHPB': 'beRNN_04',
    'sub-96WID': 'beRNN_05'
}

beRNN_brain_snip_dict = {
    'sub-SNIPKPB84': 'beRNN_01',
    'sub-SNIPYL4AS': 'beRNN_02',
    'sub-SNIP6IECX': 'beRNN_03',
    'sub-SNIPDKHPB': 'beRNN_04',
    'sub-SNIP96WID': 'beRNN_05'
}

def fisher_z(r):
    """Fisher z-transform of correlation coefficients."""
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def node_strength_vectors(fc_mats, absolute=False):
    """
    Compute 256-length connectivity-strength vector per task.

    Parameters
    ----------
    fc_mats : ndarray, shape (n_tasks, 256, 256)
        Functional correlation matrices, one per task.
    absolute : bool
        If True, use absolute correlations (sum of |r|).
        If False, use signed mean correlations.

    Returns
    -------
    vects : ndarray, shape (n_tasks, 256)
        Node-strength vectors per task.
    """
    n_tasks, n_nodes, _ = fc_mats.shape
    vects = np.zeros((n_tasks, n_nodes))
    for t in range(n_tasks):
        M = fisher_z(fc_mats[t].copy())
        np.fill_diagonal(M, 0.0)
        np.nan_to_num(M, copy=False, nan=0.0)

        # if absolute:
        #     vects[t] = np.sum(np.abs(M), axis=1)
        # else:
        vects[t] = np.mean(M, axis=1)
    # optional normalization across tasks
    vects = (vects - vects.mean(axis=1, keepdims=True)) / (vects.std(axis=1, keepdims=True) + 1e-12)
    return vects


########################################################################################################################
# head - Preprocessing brain rdm #######################################################################################
########################################################################################################################
if setup['preprocess_fMRI2rdm'] == True:
    for participant in setup['participants_snip']:
        if participant == 'sub-SNIPDKHPB': recordings = ['01', '02', '03']
        else: recordings = ['01', '02', '03', '04', '05']

        for recording in recordings:
            averageVector_list = []
            for task in setup['tasks']:
                # Call avg. functional correlation matrix
                averageVector = pd.read_csv(os.path.join(rf'W:\group_csp\in_house_datasets\bernn\mri\derivatives\xcpd_0.11.1\{participant}{recording}\func', f'{participant}{recording}_task-{task}_space-MNI152NLin2009cAsym_seg-4S256Parcels_stat-mean_timeseries.tsv'), sep='\t')
                np.nan_to_num(averageVector.values, copy=False, nan=0)
                averageVector_list.append(averageVector.values.mean(axis=0))
                averageVector_list_stacked = np.stack(averageVector_list, axis=0)

            # brain_vectors = node_strength_vectors(averageVector_list_stacked, absolute=True)

            rdm_metric = 'cosine'
            rdm, rdm_vector = clustering.compute_rdm(averageVector_list_stacked.T, rdm_metric)

            np.save(os.path.join(setup['directory_rdm'], f'{participant}_ses-{recording}_rdm_{rdm_metric}.npy'), rdm)


########################################################################################################################
# head - Correlation ###################################################################################################
########################################################################################################################
# Overall Interpretation of Topological Markers: Differences in these markers suggest the RNNs and brain networks have
# fundamentally different architectural rules and structural biases (e.g., one might be more random and efficient globally,
# while the other is more modular and efficient locally).

if setup['comparison'] == 'correlation':

    # info: brain ######################################################################################################
    topologicalMarker_dict_brain = {
        'beRNN_01': {},
        'beRNN_02': {},
        'beRNN_03': {},
        'beRNN_04': {},
        'beRNN_05': {}}

    for brain in setup['participants']:

        global_eff_list = []
        transitivity_list = []
        density_list = []
        avg_eigenvector_list = []
        avg_clustering_list = []
        mod_value_sparse_list = []
        participation_coefficient_list = []
        avg_degree_list = []
        avg_betweenness_list = []
        avg_closeness_list = []

        if beRNN_brain_dict[brain] == 'beRNN_04':
            numberOfModels = 3
        else:
            numberOfModels = 5

        for model in range(1, numberOfModels+1):
            brain_directory = fr'{setup["folder_brain"]}\functional_matrices\{brain}_ses-0{model}-avg.npy'
            matrix_brain = np.load(brain_directory)

            # averaged_correlation_matrix_thresholded = apply_threshold(matrix_brain, threshold)
            averaged_correlation_matrix_thresholded = apply_density_threshold(matrix_brain, setup['threshold'])
            np.fill_diagonal(averaged_correlation_matrix_thresholded, 0)  # prevent self-loops

            # Function to apply a threshold to the matrix
            G_brain = nx.from_numpy_array(averaged_correlation_matrix_thresholded)

            # Optionally calculate averages of node-based metrics
            global_eff = nx.global_efficiency(G_brain)
            transitivity = nx.transitivity(G_brain)
            density = nx.density(G_brain)
            global_eff_list.append(global_eff)
            transitivity_list.append(transitivity)
            density_list.append(density)

            eigenvector = nx.eigenvector_centrality_numpy(G_brain)
            avg_eigenvector = np.mean(list(eigenvector.values()))
            avg_eigenvector_list.append(avg_eigenvector)

            # clustering
            clustering = nx.clustering(G_brain)
            avg_clustering = np.mean(list(clustering.values()))
            avg_clustering_list.append(avg_clustering)
            # modularity
            communities_sparse = greedy_modularity_communities(G_brain)
            mod_value_sparse = modularity(G_brain, communities_sparse)
            mod_value_sparse_list.append(mod_value_sparse)
            # participation
            pc_dict = participation_coefficient(G_brain, communities_sparse)
            avg_pc = np.mean(list(pc_dict.values()))
            participation_coefficient_list.append(avg_pc)

            clustering = nx.clustering(G_brain)
            avg_clustering = np.mean(list(clustering.values()))
            avg_clustering_list.append(avg_clustering)

            degrees = nx.degree(G_brain) # The degree of a node in a graph is the count of edges connected to that node.
            avg_degree = np.mean(list(dict(G_brain.degree()).values()))
            avg_degree_list.append(avg_degree)

            betweenness = nx.betweenness_centrality(G_brain)
            avg_betweenness = np.mean(list(betweenness.values()))
            avg_betweenness_list.append(avg_betweenness)

            closeness = nx.closeness_centrality(G_brain) # Closeness centrality measures the average distance from a node to all other nodes in the network.
            avg_closeness = np.mean(list(closeness.values()))
            avg_closeness_list.append(avg_closeness)

        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['global_eff'] = global_eff_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['transitivity'] = transitivity_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['density'] = density_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['avg_eigenvector'] = avg_eigenvector_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['avg_clustering'] = avg_clustering_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['mod_value_sparse'] = mod_value_sparse_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['participation_coefficient'] = participation_coefficient_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['avg_degree'] = avg_degree_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['avg_betweenness'] = avg_betweenness_list
        topologicalMarker_dict_brain[beRNN_brain_dict[brain]]['avg_closeness'] = avg_closeness_list

    brain_fileDirectory = os.path.join(setup['folder_topologicalMarker_pValue_lists'])
    os.makedirs(brain_fileDirectory, exist_ok=True)
    with open(os.path.join(brain_fileDirectory, f'topologicalMarker_dict_brain_{setup["threshold"]}.json'), 'w') as fp:
        json.dump(topologicalMarker_dict_brain, fp)

    # info: beRNN ######################################################################################################
    topologicalMarker_dict_beRNN = {
        'beRNN_01': {},
        'beRNN_02': {},
        'beRNN_03': {},
        'beRNN_04': {},
        'beRNN_05': {}}

    for participant in setup['participants_beRNN']:

        folder_beRNN = setup["folder_beRNN"].replace("beRNN_01", participant, 1)
        hp_set = folder_beRNN.split('_')[-1]
        model_folder = rf'{folder_beRNN}\{setup["dataType"]}\{participant}\{hp_set}'

        global_eff_list = []
        transitivity_list = []
        density_list = []
        avg_eigenvector_list = []
        avg_clustering_list = []
        mod_value_sparse_list = []
        participation_coefficient_list = []
        avg_degree_list = []
        avg_betweenness_list = []
        avg_closeness_list = []

        # if participant == 'beRNN_04':
        #     numberOfModels = 3
        # else:
        #     numberOfModels = 5

        numberOfModels = 20

        for model in os.listdir(model_folder)[:numberOfModels]:
        # for model in os.listdir(model_folder):
            if model == 'times.txt':
                continue
            if 'fundamentals' in model_folder or 'fm' in model_folder:
                pkl_beRNN3 = rf'{model_folder}\{model}\model_month_6\var_test_lay1_rule_fundamentals.pkl'
                pkl_beRNN2 = rf'{model_folder}\{model}\model_month_6\corr_test_lay1_rule_fundamentals.pkl'
            elif 'multiTask' in model_folder or 'AllTask' in model_folder:
                pkl_beRNN3 = rf'{model_folder}\{model}\model_month_6\var_test_lay1_rule_all.pkl'
                pkl_beRNN2 = rf'{model_folder}\{model}\model_month_6\corr_test_lay1_rule_all.pkl'
            else:
                pkl_beRNN3 = rf'{model_folder}\{model}\model_month_6\var_test_lay1_rule_taskSubset.pkl'
                pkl_beRNN2 = rf'{model_folder}\{model}\model_month_6\corr_test_lay1_rule_taskSubset.pkl'

            # h_var_all as basis for thresholding dead neurons as h_corr_all can result in high values for dead neurons
            res3 = load_pickle(pkl_beRNN3)
            h_var_all_ = res3['h_var_all']
            ind_active = np.where(h_var_all_.sum(axis=1) >= setup['threshold'])[0]

            # h_corr_all as representative for modularity _analysis reflecting similar neuron behavior
            res2 = load_pickle(pkl_beRNN2)
            h_corr_all_ = res2['h_corr_all']
            h_corr_all_ = h_corr_all_.mean(axis=2)  # average over all tasks

            hp = load_hp(rf'{model_folder}\{model}\model_month_6')
            numberOfHiddenUnits = hp['n_rnn']

            if ind_active.shape[0] < h_corr_all_.shape[0] and ind_active.shape[0] < h_corr_all_.shape[1] and ind_active.shape[0] > 1:
                h_corr_all_ = h_corr_all_[ind_active, :]
                h_corr_all = h_corr_all_[:, ind_active]

                # Apply threshold
                functionalCorrelation_density = apply_density_threshold(h_corr_all, density=density)
            else:
                functionalCorrelation_density = np.zeros((numberOfHiddenUnits, numberOfHiddenUnits))  # fix: Get individual number of hidden units # Create different dummy matrix, that leads to lower realtive count

            # Compute modularity
            np.fill_diagonal(functionalCorrelation_density, 0)  # prevent self-loops

            # Function to apply a threshold to the matrix
            G_beRNN = nx.from_numpy_array(functionalCorrelation_density)

            # Optionally calculate averages of node-based metrics
            global_eff = nx.global_efficiency(G_beRNN)
            transitivity = nx.transitivity(G_beRNN)
            density = nx.density(G_beRNN)
            global_eff_list.append(global_eff)
            transitivity_list.append(transitivity)
            density_list.append(density)

            eigenvector = nx.eigenvector_centrality_numpy(G_beRNN)
            avg_eigenvector = np.mean(list(eigenvector.values()))
            avg_eigenvector_list.append(avg_eigenvector)

            # clustering
            clustering = nx.clustering(G_beRNN)
            avg_clustering = np.mean(list(clustering.values()))
            avg_clustering_list.append(avg_clustering)
            # modularity
            communities_sparse = greedy_modularity_communities(G_beRNN)
            mod_value_sparse = modularity(G_beRNN, communities_sparse)
            mod_value_sparse_list.append(mod_value_sparse)
            # participation
            pc_dict = participation_coefficient(G_beRNN, communities_sparse)
            avg_pc = np.mean(list(pc_dict.values()))
            participation_coefficient_list.append(avg_pc)

            degrees = nx.degree(G_beRNN)  # The degree of a node in a graph is the count of edges connected to that node.
            avg_degree = np.mean(list(dict(G_beRNN.degree()).values()))
            avg_degree_list.append(avg_degree)

            betweenness = nx.betweenness_centrality(G_beRNN)
            avg_betweenness = np.mean(list(betweenness.values()))
            avg_betweenness_list.append(avg_betweenness)

            closeness = nx.closeness_centrality(G_beRNN)  # Closeness centrality measures the average distance from a node to all other nodes in the network.
            avg_closeness = np.mean(list(closeness.values()))
            avg_closeness_list.append(avg_closeness)

        topologicalMarker_dict_beRNN[participant]['global_eff'] = global_eff_list
        topologicalMarker_dict_beRNN[participant]['transitivity'] = transitivity_list
        topologicalMarker_dict_beRNN[participant]['density'] = density_list
        topologicalMarker_dict_beRNN[participant]['avg_eigenvector'] = avg_eigenvector_list
        topologicalMarker_dict_beRNN[participant]['avg_clustering'] = avg_clustering_list
        topologicalMarker_dict_beRNN[participant]['mod_value_sparse'] = mod_value_sparse_list
        topologicalMarker_dict_beRNN[participant]['participation_coefficient'] = participation_coefficient_list
        topologicalMarker_dict_beRNN[participant]['avg_degree'] = avg_degree_list
        topologicalMarker_dict_beRNN[participant]['avg_betweenness'] = avg_betweenness_list
        topologicalMarker_dict_beRNN[participant]['avg_closeness'] = avg_closeness_list

    beRNN_fileDirectory = os.path.join(setup['folder_topologicalMarker_pValue_lists'], 'allModels')
    os.makedirs(beRNN_fileDirectory, exist_ok=True)
    current_folder_beRNN = setup["folder_beRNN"].split('\\')[-1]
    with open(os.path.join(beRNN_fileDirectory, f'topologicalMarker_dict_beRNN_{current_folder_beRNN}_{setup["threshold"]}.json'), 'w') as fp:
        json.dump(topologicalMarker_dict_beRNN, fp)


########################################################################################################################
# head - RSA ###########################################################################################################
########################################################################################################################
# Overall Interpretation of RDA: Differences in RDA results suggest that the two network types encode information differently;
# the patterns of neural activity discriminate between stimuli in distinct ways, even if the underlying structural topology
# (measured by the markers above) might seem similar. RDA speaks to the functional computation being performed

elif setup['comparison'] == 'rsa':
    # info: brain ######################################################################################################
    # matrix_brain = r'W:\group_csp\analyses\oliver.frank\share\functional_matrices_rdm\rsa_dissim.npy'
    directory_brain = rf'{setup["folder_brain"]}\functional_matrices_rdm'

    def ascendingNumbers_brain(e):
        return int(e.split('ses-0')[1].split('_')[0])

    def vec(rdm):  # → length 66 (for 12 tasks)
        idx = np.triu_indices_from(rdm, k=1)
        return rdm[idx]

    rdm_dict_brain = {}
    for brain in setup['participants_snip']:
        # rdmFiles = [i for i in os.listdir(str(directory_).format(brain=brain)) if i.endswith('.npy')]
        rdmFiles = [i for i in os.listdir(directory_brain) if brain in i and i.endswith('.npy')]
        rdmFiles.sort(key=ascendingNumbers_brain)  # Sort list according to information chunk given in key function
        rdm_dict_brain[brain] = rdmFiles

    # Load all .npy files as ndarrays and save them in already existing dict
    for brain in setup['participants_snip']:

        # info: Only have 3 brainImages from beRNN_04
        if beRNN_brain_snip_dict[brain] == 'beRNN_04':
            numberOfModels = 3
        else:
            numberOfModels = 5

        for rdm in range(0, numberOfModels):
            rdm_dict_brain[brain][rdm] = np.load(Path(directory_brain, rdm_dict_brain[brain][rdm]))

    # Due to feasibility rdm vector creation was outsourced from compute_rdm() - function vec() does the exact same
    rdm_vec_dict_brain = {s: [vec(r) for r in rdm_list] for s, rdm_list in rdm_dict_brain.items()} # info: Upper triangle is created here

    # Compute spearman rank correlation for each possible combination of model groups (within/between)
    subjects_brain = list(rdm_vec_dict_brain.keys()) # fix should be the same


    # info: beRNN ######################################################################################################
    rdm_dict_beRNN = {}

    for participant in setup['participants_beRNN']:

        folder_beRNN = setup["folder_beRNN"].replace("beRNN_01", participant, 1)
        model_folder = rf'{folder_beRNN}\{setup["dataType"]}\{participant}\{folder_beRNN.split("_")[-1]}'

        rdmList = []

        # info: Only have 3 brainImages from beRNN_04
        if participant == 'beRNN_04':
            numberOfModels = setup['numberOfModels'][1]
        else:
            numberOfModels = setup['numberOfModels'][0]

        for modelNumber in range(0, numberOfModels):
            model = os.listdir(model_folder)[modelNumber]
            if model == 'times.txt':
                continue

            if 'fundamentals' in model_folder or 'fm' in model_folder:
                pkl_beRNN = rf'{model_folder}\{model}\model_month_6\mean_test_lay1_rule_fundamentals.pkl'
            elif 'domainTask' in model_folder:
                pkl_beRNN = rf'{model_folder}\{model}\model_month_6\mean_test_lay1_rule_taskSubset.pkl'
            elif 'multiTask' in model_folder or 'AllTask' in model_folder:
                pkl_beRNN = rf'{model_folder}\{model}\model_month_6\mean_test_lay1_rule_all.pkl'
            else:
                print('No .pkl file found!')

            meanMatrix_taskwise = load_pickle(pkl_beRNN)

            h_mean_all = meanMatrix_taskwise['h_mean_all']
            rdm, rdm_vector = clustering.compute_rdm(h_mean_all, 'cosine')
            rdmList.append(rdm)

        rdm_dict_beRNN[participant] = rdmList
        # Due to feasibility rdm vector creation was outsourced from compute_rdm() - function vec() does the exact same
        rdm_vec_dict_beRNN = {s: [vec(r) for r in rdm_list] for s, rdm_list in rdm_dict_beRNN.items()}
        subjects_beRNN = list(rdm_vec_dict_beRNN.keys())  # fix should be the same

        # fix: Are they sorted??
        # def ascendingNumbers(e):
        #     return int(e.split('_')[0])
        # rdmFiles.sort(key=ascendingNumbers)  # Sort list according to information chunk given in key function
        # rdm_dict[participant] = rdmFiles

    # RSA comparison ###################################################################################################

    # First part of comparison variable definition
    if setup['modalityWithin_comparison'] == 'brain':
        subjects_beRNN = subjects_brain
        rdm_vec_dict_beRNN = rdm_vec_dict_brain
        print('brain-brain comparison')
    elif setup['modalityWithin_comparison'] == 'beRNN':
        subjects_brain = subjects_beRNN
        rdm_vec_dict_brain = rdm_vec_dict_beRNN
        print('beRNN-beRNN comparison')
    else:
        print('brain-beRNN comparison')
    within_rsa = {s: [] for s in subjects_beRNN}
    between_rsa = {s: [] for s in subjects_beRNN}

    # Loop over all combinations and assign results to the right dict (within/between)
    for i, s1 in enumerate(subjects_beRNN):
        for j, s2 in enumerate(subjects_brain):
            for v1 in rdm_vec_dict_beRNN[s1]:
                for v2 in rdm_vec_dict_brain[s2]:

                    rho = scipy.stats.spearmanr(v1, v2).correlation
                    # Second part of comparison variable definition
                    if setup['modalityWithin_comparison'] == 'standard':
                        if s1 == beRNN_brain_snip_dict[s2]:
                            within_rsa[s1].append(rho)
                        else:
                            between_rsa[s1].append(rho)
                    else:
                        if s1 == s2:
                            within_rsa[s1].append(rho)
                        else:
                            between_rsa[s1].append(rho)

    # Create dict for each comparison's mean value
    within_mean = np.array([np.mean(within_rsa[s]) for s in subjects_beRNN])  # shape (5,)
    between_mean = np.array([np.mean(between_rsa[s]) for s in subjects_beRNN])  # shape (5,)

    z_within = np.arctanh(within_mean)  # Fisher z
    z_between = np.arctanh(between_mean)

    t, p = ttest_rel(z_within, z_between)  # H₀: means equal
    print(f'Within  mean ρ = {within_mean.mean():.3f}')
    print(f'Between mean ρ = {between_mean.mean():.3f}')
    print(f'Paired t(4) = {t:.2f}, p = {p:.4f}')

    # Create ordered list of vectors and labels
    all_vecs_beRNN = []
    labels_beRNN = []
    for subj in subjects_beRNN:
        for i, vec in enumerate(rdm_vec_dict_beRNN[subj], 1):
            all_vecs_beRNN.append(vec)
            labels_beRNN.append(f"beRNN_{subj}_M{i}")

    # Create ordered list of vectors and labels
    all_vecs_brain = []
    labels_brain = []
    for subj in subjects_brain:
        for i, vec in enumerate(rdm_vec_dict_brain[subj], 1):
            all_vecs_brain.append(vec)
            try:
                labels_brain.append(f"brain_{beRNN_brain_snip_dict[subj]}_M{i}")
            except:
                labels_brain.append(f"beRNN_{subj}_M{i}") # brain conversion to beRNN labeling
                
    # Third part of comparison variable definition
    if setup['modalityWithin_comparison'] == 'brain':
        all_vecs_beRNN = all_vecs_brain
        labels_beRNN = labels_brain
        modelNames = ['brain', 'brain']
    elif setup['modalityWithin_comparison'] == 'beRNN':
        all_vecs_brain = all_vecs_beRNN
        labels_brain = labels_beRNN
        modelNames = ['beRNN', 'beRNN']
    else:
        modelNames = ['beRNN', 'brain']

    # Compute full Spearman RSA similarity matrix
    n_models_beRNN = len(all_vecs_beRNN)
    n_models_brain = len(all_vecs_brain)
    rsa_sim = np.zeros((n_models_beRNN, n_models_brain))
    for i in range(n_models_beRNN):
        for j in range(n_models_brain):
            rho, _ = spearmanr(all_vecs_beRNN[i], all_vecs_brain[j])
            rsa_sim[i, j] = rho

    # Convert to dissimilarity
    rsa_dissim = 1 - rsa_sim

    # Plot RSA heatmap
    plt.figure(figsize=(10, 10))
    ax = sns.heatmap(
        rsa_dissim,
        cmap="viridis",
        xticklabels=labels_beRNN,
        yticklabels=labels_brain,
        square=True,
        linewidths=0,
        cbar_kws={
            'shrink': 0.5,
            'aspect': 10,
            'label': 'RSA',
            'ticks': np.linspace(0, 0.5, 6)
        },
        vmin=0, vmax=0.5
    )

    # ----- SET CUSTOM TICKS -----
    n_groups = len(subjects_beRNN)
    # group_size = 20  # number of models per participant
    # tick_positions = [i * group_size + group_size // 2 for i in range(n_groups)]
    tick_positions = []

    # Draw rectangles around diagonal blocks
    for g, subject in enumerate(subjects_beRNN):

        if subject == 'beRNN_04' and setup['modalityWithin_comparison'] != 'beRNN':
            group_size = setup['numberOfModels'][1]
            tick_positions.append((g * group_size + group_size // 2) + 6.5)
            start = (g * group_size)+6
        elif subject == 'beRNN_05' and setup['modalityWithin_comparison'] != 'beRNN':
            group_size = setup['numberOfModels'][0]
            tick_positions.append((g * group_size + group_size // 2) - 1.5)
            start = (g * group_size)-2
        else:
            group_size = setup['numberOfModels'][0]
            tick_positions.append((g * group_size + group_size // 2) + 0.5)
            start = g * group_size


        rect = patches.Rectangle(
            (start, start),  # (x, y) start position
            group_size,  # width
            group_size,  # height
            fill=False,  # no fill
            edgecolor='lightsteelblue',  # border color
            linewidth=3  # line thickness
        )
        ax.add_patch(rect)

    # Create text lines
    stats_text = (
        f"Within mean ρ = {within_mean.mean():.3f}\n"
        f"Between mean ρ = {between_mean.mean():.3f}\n"
        f"Paired t(4) = {t:.2f}, p = {p:.4f}"
    )

    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)

    # Colorbar font sizes
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('RDA (1 - Spearman ρ)', fontsize=12)

    # ax.set_xticklabels(subjects_beRNN, rotation=0, fontsize=12)
    # ax.set_yticklabels(subjects_brain, rotation=90, fontsize=12)

    ax.set_xticklabels(setup['paper_nomenclatur'], rotation=0, fontsize=12)
    ax.set_yticklabels(setup['paper_nomenclatur'], rotation=90, fontsize=12)

    plt.title("RDA matrix", fontsize=20, fontweight='bold')
    plt.xlabel("Participant", fontsize=16)
    plt.ylabel("Participant", fontsize=16)
    plt.tight_layout()

    plt.subplots_adjust(bottom=0.2)  # space for stats

    plt.figtext(
        0.435, 0.15,
        stats_text,
        ha='center', va='top',
        fontsize=14,
        linespacing=1.5
    )

    os.makedirs(setup['rsa_directory'], exist_ok=True)
    plt.savefig(os.path.join(setup['rsa_directory'], rf'RDAmatrix-{os.path.basename(setup["folder_beRNN"])}-{setup["modalityWithin_comparison"]}.png'), bbox_inches='tight', dpi=300)

    plt.show()


########################################################################################################################
# head - Visualize networks seperated ##################################################################################
########################################################################################################################

if setup["visualize_fMRI"] == True:
    directory_fMRI = rf'{setup["folder_brain"]}\functional_matrices'

    for participant in setup["participants"]:
        functionalMatrix_faces_ = np.load(os.path.join(directory_fMRI, f'{participant}_ses-{setup["session"]}-task-faces-atlas-4S256Parcels.npy'))
        functionalMatrix_faces = np.nan_to_num(functionalMatrix_faces_)
        # plt.imshow(functionalMatrix_faces)
        # plt.title('faces')
        # plt.show()

        functionalMatrix_flanker_ = np.load(os.path.join(directory_fMRI, f'{participant}_ses-{setup["session"]}-task-flanker-atlas-4S256Parcels.npy'))
        functionalMatrix_flanker = np.nan_to_num(functionalMatrix_flanker_)
        # plt.imshow(functionalMatrix_flanker)
        # plt.title('flanker')
        # plt.show()

        functionalMatrix_nback_ = np.load(os.path.join(directory_fMRI, f'{participant}_ses-{setup["session"]}-task-nback-atlas-4S256Parcels.npy'))
        functionalMatrix_nback = np.nan_to_num(functionalMatrix_nback_)
        # plt.imshow(functionalMatrix_nback)
        # plt.title('nback')
        # plt.show()

        functionalMatrix_reward_ = np.load(os.path.join(directory_fMRI, f'{participant}_ses-{setup["session"]}-task-reward-atlas-4S256Parcels.npy'))
        functionalMatrix_reward = np.nan_to_num(functionalMatrix_reward_)
        # plt.imshow(functionalMatrix_reward)
        # plt.title('reward')
        # plt.show()

        functionalMatrix_mean = np.mean([functionalMatrix_faces, functionalMatrix_flanker, functionalMatrix_nback, functionalMatrix_reward], axis=0)
        plt.imshow(functionalMatrix_mean)
        plt.title('mean')
        plt.show()


if setup["correlationOf_correlationMatices_fMRI"] == True:
    # Functional correlation matrices - Pearson Correlation

    correlationMatrices = []
    for participant in setup["participants"]:

        if participant == 'sub-DKHPB':
            recordings = ['01', '02', '03']
        else:
            recordings = ['01', '02', '03', '04', '05']

        for recording in recordings:
            print(participant, recording)
            # Load correlation matrices
            path = os.path.join(setup["folder_brain"], 'functional_matrices', f'{participant}_ses-{recording}-avg.npy')
            correlationMatrix = np.load(path)
            correlationMatrices.append(correlationMatrix)

    # Function to extract upper triangle (excluding diagonal)
    def upper_triangle(matrix):
        return matrix[np.triu_indices_from(matrix, k=1)]

    # Extract flattened upper triangles
    flattened_correlationMatrices = [upper_triangle(mat) for mat in correlationMatrices]

    # Compute pairwise correlations
    # n = len(recordings)
    n = 23
    corr_matrix_of_corrs = np.zeros((n, n))

    for (i, vec1), (j, vec2) in itertools.combinations(enumerate(flattened_correlationMatrices), 2):
        corr, _ = pearsonr(vec1, vec2)
        corr_matrix_of_corrs[i, j] = corr
        corr_matrix_of_corrs[j, i] = corr  # symmetric

    np.fill_diagonal(corr_matrix_of_corrs, 1)  # self-correlation

    plt.figure()
    plt.imshow(corr_matrix_of_corrs, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"Pearson Similarity")
    plt.xlabel("model")
    plt.ylabel("model")
    plt.show()


if setup["correlationOf_correlationMatices_beRNN"] == True:
    # Functional correlation matrices - Pearson Correlation

    correlationMatrices = []
    for participant in setup["participants_beRNN"]:
        folder_beRNN = setup["folder_beRNN"].replace("beRNN_01", participant, 1)
        modelPath = os.path.join(folder_beRNN, setup["dataType"], participant, folder_beRNN.split("_")[-1])
        modelList = os.listdir(modelPath)

        # if participant == 'beRNN_04':
        #     numberOfModels = 3
        # else:
        numberOfModels = 5

        for modelNumber in range(0, numberOfModels):
            # Load correlation matrices
            if 'fundamentals' in modelPath or 'fm' in modelPath:
                correlationMatrix_ = load_pickle(rf'{modelPath}\{modelList[modelNumber]}\model_month_6\corr_test_lay1_rule_fundamentals.pkl')
            elif 'multiTask' in modelPath or 'AllTask' in modelPath:
                correlationMatrix_ = load_pickle(rf'{modelPath}\{modelList[modelNumber]}\model_month_6\corr_test_lay1_rule_all.pkl')
            else:
                correlationMatrix_ = load_pickle(rf'{modelPath}\{modelList[modelNumber]}\model_month_6\corr_test_lay1_rule_taskSubset.pkl')

            correlationMatrix = correlationMatrix_['h_corr_all'].mean(axis=2)
            correlationMatrices.append(correlationMatrix)

    # Function to extract upper triangle (excluding diagonal)
    def upper_triangle(matrix):
        return matrix[np.triu_indices_from(matrix, k=1)]

    # Extract flattened upper triangles
    flattened_correlationMatrices = [upper_triangle(mat) for mat in correlationMatrices]

    # Compute pairwise correlations
    # n = len(recordings)
    n = 25
    corr_matrix_of_corrs = np.zeros((n, n))

    for (i, vec1), (j, vec2) in itertools.combinations(enumerate(flattened_correlationMatrices), 2):
        corr, _ = pearsonr(vec1, vec2)
        corr_matrix_of_corrs[i, j] = corr
        corr_matrix_of_corrs[j, i] = corr  # symmetric

    np.fill_diagonal(corr_matrix_of_corrs, 1)  # self-correlation

    plt.figure()
    plt.imshow(corr_matrix_of_corrs, cmap="vlag", vmin=-1, vmax=1)
    plt.colorbar(label="Correlation")
    plt.title(f"Correlation Matrix")
    plt.xlabel("model")
    plt.ylabel("model")
    plt.show()


if setup["visualize_dti"] == True:
    # Correlation matrices of dti correlation matrices #####################################################################
    order = ['sub-KPB84', 'sub-YL4AS', 'sub-6IECX', 'sub-DKHPB', 'sub-96WID']
    order_index = {sub: i for i, sub in enumerate(order)}

    def sort_key(name: str):
        sub_match = re.search(r"(sub-[A-Za-z0-9]+)", name)
        ses_match = re.search(r"ses-([0-9]+)", name)

        subject = sub_match.group(1) if sub_match else ""
        session = int(ses_match.group(1)) if ses_match else 0

        # Use order ranking, fallback to a large number if unknown
        rank = order_index.get(subject, 9999)

        return (rank, session)


    directory = r'W:\group_csp\analyses\oliver.frank\share\adjacency_matrices_dti'
    # directory = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNN_v1\masks\connectomes_legacy'
    dti_matrices_list = []
    dti_matrices_list_ = []

    for dti_matrix_ in os.listdir(directory):
        if 'standardized' in dti_matrix_ and 'ses-03' in dti_matrix_:
            dti_matrices_list_.append(dti_matrix_)
    # Sort created list
    dti_matrices_list_sorted_ = sorted(dti_matrices_list_, key=sort_key)

    for dti_matrix_ in dti_matrices_list_sorted_:
            dti_matrix = np.load(os.path.join(directory,dti_matrix_))
            dti_matrices_list.append(dti_matrix)

    n = len(dti_matrices_list)
    pearson_sim = np.zeros((n, n))
    cosine_sim = np.zeros((n, n))

    # plt.imshow(np.load(os.path.join(directory,dti_matrices_list_sorted_[0])))
    # plt.show()

    for v1, dti_matrix_v1 in enumerate(dti_matrices_list):
        v1_vector = dti_matrix_v1[np.triu_indices_from(dti_matrix_v1, k=1)]
        for v2, dti_matrix_v2 in enumerate(dti_matrices_list):
            v2_vector = dti_matrix_v2[np.triu_indices_from(dti_matrix_v2, k=1)]

            # Pearson correlation
            pearson_sim[v1, v2] = pearsonr(v1_vector, v2_vector)[0]

            # Cosine similarity (scipy returns cosine *distance*, so invert)
            cosine_sim[v1, v2] = cosine(v1_vector, v2_vector) # calculates dissimilarity by default

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    sns.heatmap(pearson_sim, vmin=0, vmax=1, cmap="viridis")
    plt.title("Pearson Similarity")

    plt.subplot(1,2,2)
    sns.heatmap(cosine_sim, vmin=0, vmax=1, cmap="viridis")
    plt.title("Cosine Disimilarity")
    plt.show()


if setup["plotTopMarkerOverDensities"] == True:
    # Visualize topological marker for different desnitites and data sets/modalities
    participant_to_paper = dict(zip(setup['participants_beRNN'], setup['paper_nomenclatur']))

    densities = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    for density in densities:
        plot_folder = 'densityPlots_multiTask_beRNN_highDimCorrects_20Models'
        directory = r'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists\allModels'
        file_path = os.path.join(directory, f'topologicalMarker_dict_beRNN__robustnessTest_multiTask_beRNN_01_highDimCorrects_256_hp_2_{density}.json')

        with open(file_path, 'r') as f:
            data = json.load(f)

        metrics = {
            "mod_value_sparse": "Modularity",
            "participation_coefficient": "Participation",
            "avg_clustering": "Clustering"
        }

        plot_data = []

        for participant, values in data.items():
            for json_key, clean_name in metrics.items():
                subset = values[json_key]
                for val in subset:
                    plot_data.append({
                        "Participant": participant,
                        "Metric": clean_name,
                        "Value": val
                    })

        df = pd.DataFrame(plot_data)

        df["Participant"] = pd.Categorical(df["Participant"],categories=setup['participants_beRNN'],ordered=True)

        # Definition der Farben
        custom_colors = {
            "beRNN_03": "#1f77b4",  # Deep Blue
            "beRNN_04": "#a1c9ed",  # Light Blue
            "beRNN_01": "#ff7f0e",  # Orange
            "beRNN_02": "#2ca02c",  # Green
            "beRNN_05": "#9467bd"  # Purple
        }

        plt.figure(figsize=(5,7))
        sns.set_style("whitegrid")

        # 1. Boxplot - Ohne 'width' für bessere Zentrierung
        sns.boxplot(
            data=df,
            x="Metric",
            y="Value",
            hue="Participant",
            palette=custom_colors,
            showfliers=False,
            zorder=1,
            boxprops=dict(linewidth=0),
            whiskerprops=dict(linewidth=0),
            capprops=dict(linewidth=0),
            medianprops=dict(linewidth=0)
        )

        # 2. Stripplot - Jitter=False zwingt die Punkte auf die Mittelachse
        sns.stripplot(
            data=df,
            x="Metric",
            y="Value",
            hue="Participant",
            palette=custom_colors,
            dodge=True,
            jitter=False,
            size=6,
            alpha=0.5,
            linewidth=0,  # ← no outline
            edgecolor="none",  # ← no outline
            zorder=2
        )

        ax = plt.gca()
        ax.yaxis.grid(True, linestyle='--', linewidth=0.8, alpha=0.6)
        ax.xaxis.grid(False)

        # Legende bereinigen
        handles, labels = plt.gca().get_legend_handles_labels()
        n_participants = df['Participant'].nunique()
        new_labels = [participant_to_paper[l] for l in labels[:n_participants]]

        plt.legend(handles[0:n_participants], new_labels,
                   bbox_to_anchor=(1.05, 1), loc='upper left', title="Participants")

        plt.ylabel("Marker Value")
        plt.ylim(-0.2, 1.1)
        plt.yticks(np.arange(-0.2, 1.1, 0.1))
        plt.tight_layout()

        saveDirectory = os.path.join(directory, plot_folder)
        os.makedirs(saveDirectory, exist_ok=True)
        plt.savefig(os.path.join(saveDirectory, f'topologicalMarker_density_{density}.png'))

        plt.show()


