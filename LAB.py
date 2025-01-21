import os
import numpy as np
import json

# Open the preprocessed datasets to check quality
directory = 'C:\\Users\\oliver.frank\\Desktop\\BackUp\\Data'
participant = 'beRNN_03'
dataClass = 'data_highDim'
task = 'RP'

fullDirectory = os.path.join(directory, participant, dataClass, task)

file = 'BeRNN_03-month_1-batch_0-EF-task_2p6f'

x = np.load(os.path.join(fullDirectory, file + '-Input.npy'))  # Input
y = np.load(os.path.join(fullDirectory, file + '-Output.npy'))  # Participant Response
y_loc = np.load(os.path.join(fullDirectory, file + '-yLoc.npy'))  # Ground Truth

with open(os.path.join(fullDirectory, file + '-Meta.json')) as f:
    meta = json.load(f) # state questions


