import pandas as pd
import numpy as np
import json
import os
import statsmodels.formula.api as smf
from scipy.stats import chi2
# import statsmodels.api as sm

topMarker_dict = {
    'mod_value_sparse': 'modularity',
    'avg_clustering': 'clustering',
    'avg_eigenvector': 'eigenvector',
    'avg_betweenness': 'betweenness',
    'avg_closeness': 'closeness'
}

optimizers = ['lbfgs', 'bfgs', 'cg', 'powell', 'nm']

density = '0.5'
files = [f'topologicalMarker_dict_beRNN__robustnessTest_fundamentals_participant_highDim_256_hp_2_{density}',
         f'topologicalMarker_dict_beRNN__robustnessTest_fundamentals_participant_highDimCorrects_256_hp_2_{density}',
        f'topologicalMarker_dict_beRNN__robustnessTest_fm_participant_highDimCorrects_256_bM_hp_2_{density}',
         f'topologicalMarker_dict_brain-{density}']

folder = rf'C:\Users\oliver.frank\Desktop\PyProjects\beRNNmodels\__topologicalMarker_pValue_lists'

# Create a list to hold dataframes from all sets
all_data_frames = []

for i, basename in enumerate(files):
    print(basename)
    file_path = os.path.join(folder, basename + '.json')

    try:
        with open(file_path, "r") as f:
            data_dict = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}. Skipping.")
        continue

    rows = []
    # 'data_dict.items()' gives us participant_id keys and marker values
    for p_id_in_set, markers in data_dict.items():

        # Iterate through the 5 marker types
        for marker_name_raw, values_list in markers.items():
            if marker_name_raw in topMarker_dict: # only take defined top Markers
                # Iterate through the 20 model values
                for value in values_list[:5]:
                    rows.append([
                        f'Set_{i + 1}',  # Add a unique Set ID (e.g., Set_1)
                        p_id_in_set,  # The original participant ID from the dict
                        topMarker_dict.get(marker_name_raw, marker_name_raw),  # Clean marker name
                        value  # The actual value
                    ])

    # Create DF for this set and add to master list
    df_set = pd.DataFrame(rows, columns=["Set_ID", "Participant_ID_Original", "Marker_Name", "Value"])
    all_data_frames.append(df_set)

# Concatenate all DataFrames into one master DataFrame
master_df = pd.concat(all_data_frames, ignore_index=True)

# Important: Create a globally unique Participant ID for the mixed model groups
# This ensures statsmodels knows 'P1' in Set 1 is different from 'P1' in Set 2
master_df['Global_Participant_ID'] = master_df['Set_ID'] + '_' + master_df['Participant_ID_Original']




# ########################################################################################################################
# # --- Select a Single Marker to Model First ---
# target_marker = 'closeness'  # Change this to 'clustering', 'eigenvector', etc.
# target_set = 'Set_4'
# # Filter the master DF for just the rows we need for this specific run
# df_single_set_single_marker = master_df[
#     (master_df['Marker_Name'] == target_marker) &
#     (master_df['Set_ID'] == target_set)]
#
# # 1. Fit the set-specific (Mixed) Model
# model_mixed = smf.mixedlm("Value ~ 1", df_single_set_single_marker, groups=df_single_set_single_marker['Participant_ID_Original'])
# for opt_method in optimizers:
#     try:
#         try:
#             # Try fitting with a specific optimizer
#             result_mixed = model_mixed.fit(method=opt_method, reml=False)
#             print(f"  Mixed Model fitted successfully with method: {opt_method} and reml:False")
#             break  # Exit the optimizer loop if successful
#         except:
#             # Try fitting with a specific optimizer
#             result_mixed = model_mixed.fit(method=opt_method, reml=True)
#             print(f"  Mixed Model fitted successfully with method: {opt_method} and reml:True")
#             break  # Exit the optimizer loop if successful
#     except (np.linalg.LinAlgError, ValueError) as e:
#         print(f"  Mixed Model fit failed with method {opt_method}. Trying next...")
#
# print(result_mixed.summary()) # goodness of fit??
#
# llf_mixed = result_mixed.llf # Log-Likelihood of the mixed model
# # 2. Fit the Null (OLS/Linear) Model
# model_ols = smf.ols("Value ~ 1", df_single_set_single_marker).fit()
# llf_ols = model_ols.llf # Log-Likelihood of the OLS model
#
# # 3. Perform the Likelihood Ratio Test
# lr_statistic = 2 * (llf_mixed - llf_ols)
#
# # The degrees of freedom for this test is 1 (we are adding one variance parameter)
# df_lrt = 1
#
# # Calculate the P-value
# p_value = chi2.sf(lr_statistic, df_lrt)
# # Deviation of fitted model distribution from theoretical null model - if significant the modelling of random effects
# # has significant influence on the explanation of variance in the data, therefore we have significant differences between
# # groups, more than within
#
# print(f"\n--- Likelihood Ratio Test Results for {target_set}, {target_marker} ---")
# print(f"LR Statistic: {lr_statistic:.4f}")
# print(f"P-value: {p_value:.4f}")
#
# # Interpretation:
# if p_value < 0.05:
#     print("Result: Statistically Significant.")
#     print("Conclusion: The 'Between-Participant' variance is significantly greater than zero (i.e., participants differ significantly from each other).")
# else:
#     print("Result: Not Statistically Significant (at p=0.05 level).")
#     print("Conclusion: We do not have enough evidence to say that participants differ significantly from each other. The variance estimates might just be due to chance variation.")




########################################################################################################################
# --- Define the parameters for the comparison ---
target_marker = 'modularity'

set_A_id = 'Set_1'
set_B_id = 'Set_4'

# --- Filter the data for only the two sets and one marker of interest ---
df_comparison = master_df[
    (master_df['Marker_Name'] == target_marker) &
    (master_df['Set_ID'].isin([set_A_id, set_B_id]))
].copy()

print(f"\n--- LRT: Comparing average {target_marker} between {set_A_id} and {set_B_id} ---")

# ------------------------------------------------------------------------------------------------
# NULL MODEL: no dataset effect, random intercept for participant
#   Value ~ 1 + (1 | Participant)
# ------------------------------------------------------------------------------------------------
model_null = smf.mixedlm(
    "Value ~ 1",
    df_comparison,
    groups=df_comparison["Global_Participant_ID"]
)

result_null = None
for opt_method in optimizers:
    try:
        result_null = model_null.fit(method=opt_method, reml=False)
        print(f"  Null model fitted with method: {opt_method}")
        break
    except (np.linalg.LinAlgError, ValueError):
        pass

# ------------------------------------------------------------------------------------------------
# FULL MODEL: dataset effect + same random intercept
#   Value ~ C(Set_ID) + (1 | Participant)
# ------------------------------------------------------------------------------------------------
model_full = smf.mixedlm(
    "Value ~ C(Set_ID)",
    df_comparison,
    groups=df_comparison["Global_Participant_ID"]
)

result_full = None
for opt_method in optimizers:
    try:
        result_full = model_full.fit(method=opt_method, reml=False)
        print(f"  Full model fitted with method: {opt_method}")
        break
    except (np.linalg.LinAlgError, ValueError):
        pass

# ------------------------------------------------------------------------------------------------
# Likelihood Ratio Test
# ------------------------------------------------------------------------------------------------
if (result_null is None) or (result_full is None):
    print("FATAL ERROR: One or both models failed to converge.")
else:
    lr_stat = 2 * (result_full.llf - result_null.llf)
    p_value = chi2.sf(lr_stat, df=1)

    print("\n--- Likelihood Ratio Test ---")
    print(f"LR statistic: {lr_stat:.4f}")
    print(f"P-value:      {p_value:.4f}")

    if p_value < 0.05:
        print("Result: Statistically significant dataset difference.")
    else:
        print("Result: No evidence for a dataset difference.")

    # Optional: still print full model for effect size reporting
    print("\n--- Full Model Summary (for effect size) ---")
    print(result_full.summary())


