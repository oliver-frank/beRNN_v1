import numpy as np

# Read vertically
# Calculate correlation over time points; alternatively: repeated measure correlation
# Demographic variables
age = [22, 23, 21, 31, 33]
sex = [0, 0, 0, 0, 1] # 0: female ; 1: male
profession = [2, 2, 2, 3, 1] # 0: unemployed ; 1: hospital care ; 2: student ; 3: employed ; 4: self-employed
education = [0, 0, 1, 1, 0] # 0: high school ; 1: undergraduate ; 2: graduate ; 3: postgraduate


# info: 1. assesment (01, 02, 03, 04, 05) ##############################################################################
bcats_trail = [80, 37, 55, 101, 56]
bcats_symbol = [61, 95, 100, 94, 90]
bcats_animals = [35, 49, 27, 36, 25]
# z_score_cognition_bcats = (value - np.mean(bcats_trail + bcats_symbol + bcats_animals)) / np.std(bcats_trail + bcats_symbol + bcats_animals)

# MRT
nback = [58.33, 87.5, 87.5, 72.92, 43.75]
reward = [6/20, 0/20, 8/20, 4/20, 4/20] # fix create accuracy with 20 max
faces = [57/58, 57/58, 57/58, 56/58, 57/58]
flanker = [109/145, 116/145, 114/145, 112/145, 114/145]
# z_score_cognition_mrt = (value - np.mean(nback + reward + faces + flanker)) / np.std(nback + reward + faces + flanker)


# info: 2. assesment (01, 02, 03, 04, 05) ##############################################################################
# MRT
nback = [95.83, 91.66, 87.5, 83.33, 83.33]
reward = [4/20, 8/20, 6/20, 6/20, 8/20]
faces = [1, 53/58, 1, 57/58, 1]
flanker = [121/145, 114/145, 113/145, 113/145, 114/145]
# z_score_cognition_mrt = (value - np.mean(nback + reward + faces + flanker)) / np.std(nback + reward + faces + flanker)


# info: 3. assesment (01, 02, 03, 04, 05) ##############################################################################
# MRT
nback = [97.92, 91.66, 89.58, 87.5, 68.75]
reward = [10/20, 6/20, 10/20, 10/20, 4/20]
faces = [1, 54/58, 57/58, 56/58, 57/58]
flanker = [107/145, 115/145, 112/145, 111/145, 128/145]
# z_score_cognition_mrt = (value - np.mean(nback + reward + faces + flanker)) / np.std(nback + reward + faces + flanker)


# info: 4. assesment (01, 02, 03, 05) ##################################################################################
# MRT
nback = [85.42, 97.92, 100, 87.5]
reward = [8/20, 4/20, 10/20, 2/20]
faces = [57/58, 53/58, 57/58, 1]
flanker = [115/145, 116/145, 111/145, 115/145]
# z_score_cognition_mrt = (value - np.mean(nback + reward + faces + flanker)) / np.std(nback + reward + faces + flanker)


# info: 5. assesment (01, 02, 03, 05) ##################################################################################
# BCATS
bcats_trail = [72, 38, 39, 48]
bcats_symbol = [76, 93, 93, 93]
bcats_animals = [47, 36, 37, 32]
# z_score_cognition_bcats = (value - np.mean(bcats_trail + bcats_symbol + bcats_animals)) / np.std(bcats_trail + bcats_symbol + bcats_animals)

# MRT
nback = [79.17, 83.33, 97.92, 66.66]
reward = [6/20, 2/20, 8/20, 8/20]
faces = [34/58, 34/58, 1, 1]
flanker = [83/145, 63/145, 113/145, 113/145]
# z_score_cognition_mrt = (value - np.mean(nback + reward + faces + flanker)) / np.std(nback + reward + faces + flanker)


