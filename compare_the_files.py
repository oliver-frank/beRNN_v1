import os

#%%
path = "/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM"

os.chdir(path)
os.listdir()
# %%

#%%
import os 

import numpy as np 
path = "/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM"
os.chdir(path)
os.listdir()
id = "BeRNN_01-month_5-batch_0-DM-task_ujcn-"
id="BeRNN_01-month_5-batch_2-DM-task_9ivx"

targets = [val for val in os.listdir() if val.count(id)>0]

print(os.listdir())

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
input_errors_removed = [val for val in targets if val.count("Input_ERRORS_RE")>0]
input_errors_only = [val for val in targets if val.count("Input_ERRORS_ONLY")>0]
print(input_original)
print(input_errors_removed)
print(input_errors_only)


#%%

org = np.load(input_original[0])
print(org.shape)
rem = np.load(input_errors_removed[0])
only = np.load(input_errors_only[0])
 

 # Input
#                y = np.load(currentTriplet[2]) # Participant Response
 #               y_loc = np.load(currentTriplet[1]) # Ground Truth # yLoc



x# %%

# %%
print(org.shape)
print(rem.shape)
print(only.shape)



# %%
# %%

targets


# maybe remove empty files 
# %%


len([val for val in os.listdir() if val.count("ORIGIN")>0])
len([val for val in os.listdir() if val.count("REM")>0])
len([val for val in os.listdir() if val.count("ONLY")>0])



# %%



os.listdir()
# %%
os.getcwd()

# %%

path = '/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM'
os.listdir(path)

# %%

def str_before_nth(s, pattern, n):
    """ Return the part of the string before the nth occurrence of the pattern. """
    count = 0
    for i, part in enumerate(s.split(pattern)):
        #print(i, part)
        count += 1
        if count == n:
            #print(s.split(pattern)[:i+1])
            return pattern.join(s.split(pattern)[:i+1])
    return ""  # Pattern doesn't occur n times


# %%
path = '/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM'
import pandas as pd
df = pd.DataFrame({"filename": os.listdir()})
df["keys"] = df["filename"].apply(str_before_nth, args=("-", 5))
df["Counts"] = df["keys"].map(df["keys"].value_counts())



# %%
trash_path = "/Users/marcschubert/Documents/rnns/trash"

not_complete = df.loc[df["Counts"]!=15]

# %%
path = "/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM"

for filename in not_complete["filename"]:
    print(os.path.join(path,filename))
    print(os.path.join(trash_path, filename))
    os.rename(os.path.join(path,filename), 
              os.path.join(trash_path, filename)
              )
    

# %%

path = '/Users/marcschubert/Documents/rnns/Data/untouched_folders_with_changes/DM_Anti'
import pandas as pd
os.chdir(path)
df = pd.DataFrame({"filename": os.listdir()})
df["keys"] = df["filename"].apply(str_before_nth, args=("-", 5))
df["Counts"] = df["keys"].map(df["keys"].value_counts())

os.getcwd()


# %%

df["Counts"].value_counts()

not_complete = df.loc[df["Counts"]!=15]

# %%

for filename in not_complete["filename"]:
    print(os.path.join(path,filename))
    print(os.path.join(trash_path, filename))
    os.rename(os.path.join(path,filename), 
              os.path.join(trash_path, filename)
              )
# %%
# %%
