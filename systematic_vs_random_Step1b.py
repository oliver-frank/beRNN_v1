# systematic vs random_classification

# Script to count error types after 'Error_Analysis_CountErrors_Step1.py' has been run 
# then an r script can be run to calculate the perc scores per each group
# then this file can be read in to classify errors in different groups
# then we will have to go into the error-analysis edit script to edit the raw data and save it afterwards, based on the split 

import os 
import json
import pandas as pd
import matplotlib.pyplot as plt
os.getcwd()
directory = "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL_V3"
os.chdir(directory)

files = [file for file in os.listdir() if file.startswith("errors_dict")]
files
def extractCounts(filename): 
    print(filename)
    counts = {}
    with open(filename) as f: 
        data = json.load(f)
    for error_name, data_list in data.items():
        print(error_name, len(data_list))
        counts.update({error_name: len(data_list)})
    df = pd.DataFrame(list(counts.items()), columns = ["error_name", "count"])
    df["taskgroup"] = filename
    return df  


dfs = []
for filename in files: 
    df = extractCounts(filename)
    dfs.append(df)

alldata = pd.concat(dfs)
alldata.to_csv("all_error_counts.csv")




### Visualization is in the r script systematic_vs_random.R

alldata["count"].value_counts()
filtered = alldata[alldata["count"] != 0]
plt.figure()
plt.hist(filtered["count"], bins = 100)


