
#%%
import os 

import numpy as np 
path = "/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM"
os.chdir(path)
os.listdir()
id = "BeRNN_01-month_5-batch_0-DM-task_ujcn-"
targets = [val for val in os.listdir() if val.count(id)>0]

print(targets)

#%%

# ['BeRNN_01-month_5-batch_0-DM-task_ujcn-Output_ERRORS_ONLY.npy',
#  'BeRNN_01-month_5-batch_0-DM-task_ujcn-yLoc_ORIGINAL.npy', 
#  'BeRNN_01-month_5-batch_0-DM-task_ujcn-Output_ERRORS_REMOVED.npy',
#   'BeRNN_01-month_5-batch_0-DM-task_ujcn-yLoc_ERRORS_ONLY.npy',
#    'BeRNN_01-month_5-batch_0-DM-task_ujcn-Response_ORIGINAL.npy', 
#    'BeRNN_01-month_5-batch_0-DM-task_ujcn-Input_ERRORS_REMOVED.npy', 
#    'BeRNN_01-month_5-batch_0-DM-task_ujcn-Meta_ERRORS_REMOVED.json', 
#    'BeRNN_01-month_5-batch_0-DM-task_ujcn-Response_ERRORS_REMOVED.npy',
#     'BeRNN_01-month_5-batch_0-DM-task_ujcn-Meta_ERRORS_ONLY.json', 
#     'BeRNN_01-month_5-batch_0-DM-task_ujcn-Response_ERRORS_ONLY.npy', 
#     'BeRNN_01-month_5-batch_0-DM-task_ujcn-Meta_ORIGINAL.json',
#      'BeRNN_01-month_5-batch_0-DM-task_ujcn-Input_ORIGINAL.npy', 'BeRNN_01-month_5-batch_0-DM-task_ujcn-Input_ERRORS_ONLY.npy', 'BeRNN_01-month_5-batch_0-DM-task_ujcn-yLoc_ERRORS_REMOVED.npy', 'BeRNN_01-month_5-batch_0-DM-task_ujcn-Output_ORIGINAL.npy']



input_original = [val for val in targets if val.count("Input_ORIGINAL")>0]


#%%

 x = np.load(currentTriplet[0]) # Input
                y = np.load(currentTriplet[2]) # Participant Response
                y_loc = np.load(currentTriplet[1]) # Ground Truth # yLoc
