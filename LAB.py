import os
import numpy as np
import json

# Open the preprocessed datasets to check quality
directory = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data'
participant = 'beRNN_05'
dataClass = 'data_lowDim_lowCognition'
task = 'DM_Anti'

fullDirectory = os.path.join(directory, participant, dataClass, task)

file = 'BeRNN_05-month_3-batch_1-DM_Anti-task_k1jg'

x = np.load(os.path.join(fullDirectory, file + '-Input.npy'))  # Input
y = np.load(os.path.join(fullDirectory, file + '-Output.npy'))  # Participant Response
y_loc = np.load(os.path.join(fullDirectory, file + '-yLoc.npy'))  # Ground Truth

with open(os.path.join(fullDirectory, file + '-Meta.json')) as f:
    meta = json.load(f) # state questions


