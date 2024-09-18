# analyze sizes of files step 5 .py
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

os.chdir("/Users/marcschubert/Documents/rnns/SplittingDataset")


# check how large the specific datasets are.
# then split the files into further sizes


def getFileDimensions(path):
    data = np.load(path, allow_pickle=True)
    data.shape
    #print(data.shape)
    try: 
        val2 = data.shape[2]
    except:
        val2 = -1
    return [data.shape[0],data.shape[1], val2] 
        


#### Run


directory = "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL"
folders = [folder for folder in os.listdir(directory) if os.path.isdir(os.path.join(directory,folder))]



shapes = []
for idx, folder in enumerate(folders):
    print(idx,folder)
    files = os.listdir(os.path.join(directory,folder))
    files = [file for file in files if file.endswith(".npy") ]
    for file_idx, file in enumerate(files):
        #print(file)
        full_path = os.path.join(directory,folder,file)
        dims = getFileDimensions(full_path)
        shape = {
            "taskgroup":folder,
            "file":file,
            "shape_0":dims[0],
            "shape_1":dims[1],
            "shape_2":dims[2],
         }
        shapes.append(shape)

df = pd.DataFrame(shapes)
df["response"] = df["file"].str.count("Response")
df["taskgroup"] = df["file"].apply(lambda file: file.split("-")[3])
df["extendedname"] = df["file"].apply(lambda file: file.split("-")[5])

resp = df[df["response"]>0]

plt.figure(figsize=(10, 6))

color_map = plt.cm.get_cmap('tab20', len(unique_taskgroups))  

data = [resp[resp['taskgroup'] == name]['shape_1'] for name in unique_taskgroups]

plt.hist(data, bins=40, stacked=True, color=[color_map(i) for i in range(len(unique_taskgroups))], label=unique_taskgroups, alpha=0.7)

plt.legend(title='Task Group')
plt.xlabel('Shape_1')
plt.ylabel('Frequency')
plt.title('Histogram Shape_1 Colored by Task Group')

plt.savefig("task group_histogram.png")
plt.show()


resp["shape_1"].value_counts()




