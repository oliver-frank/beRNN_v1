#%%

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



# %%

def load_json(path):
    with open(path) as file:
        data = json.load(file)
        print("JSON loded.")
        return data




def extractMetrics(data, metric):
    dfs = []
    keys = data.keys()
    metric_keys = [key for key in keys if key.startswith(metric) ]
    metric_keys = [key for key in metric_keys if key.find("train") < 0]
    print(metric_keys)
    for k in metric_keys:
        extracted_data = data[k]
        print(len(extracted_data))
        df = pd.DataFrame({
            "trials":data["trials"],
            "times":data["times"],
            "group":k, 
                           
                           "values":extracted_data })
        dfs.append(df)
    all = pd.concat(dfs)
    return all




#%%

os.chdir("/Users/marcschubert/Documents/rnns/models")
# XModel_150_BeRNN_01_Month_2-6_0.1Threshold

log_files = []
for root, dirs, files in os.walk("/Users/marcschubert/Documents/rnns/models/XModel_150_BeRNN_01_Month_2-6_0.1Threshold"):
    for file in files:
        if file == 'log.json':
            print(file)
            log_files.append(os.path.join(root,file))



# %%


log_files
# %%





for path in log_files:
    training_set = path.split("/")[7]
    print(training_set)

    data = load_json(path)

    metric = "perf"
    bound  = extractMetrics(data, metric=metric)
    bound.to_csv(training_set+ "_" + metric + ".csv")
    fig, ax = plt.subplots()
    for label, group_df in bound.groupby("group"):
        ax.plot(group_df["trials"], group_df["values"], label = label)

    ax.set_xlabel("Trials")
    ax.set_ylabel("Values")
    ax.set_title(training_set +" - "+ metric)
    ax.legend()
    plt.savefig(training_set +" - "+ metric + ".png")
    plt.show()


    metric = "creg"
    bound  = extractMetrics(data, metric=metric)
    bound.to_csv(training_set+ "_" + metric + ".csv")
    fig, ax = plt.subplots()
    for label, group_df in bound.groupby("group"):
        ax.plot(group_df["trials"], group_df["values"], label = label)

    ax.set_xlabel("Trials")
    ax.set_ylabel("Values")
    ax.set_ylim(ymax=0.1)
    ax.set_title(training_set +" - "+ metric)
    ax.legend()
    plt.savefig(training_set +" - "+ metric + ".png")
    plt.show()

    metric = "cost"
    bound  = extractMetrics(data, metric=metric)
    bound.to_csv(training_set+ "_" + metric + ".csv")
    fig, ax = plt.subplots()
    for label, group_df in bound.groupby("group"):
        ax.plot(group_df["trials"], group_df["values"], label = label)

    ax.set_xlabel("Trials")
    ax.set_ylabel("Values")
    ax.set_ylim(ymax=1)
    ax.set_title(training_set +" - "+ metric)
    ax.legend()
    plt.savefig(training_set +" - "+ metric + ".png")
    plt.show()




print("done")