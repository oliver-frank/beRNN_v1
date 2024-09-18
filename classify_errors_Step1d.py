import pandas as pd 
import json
# path to error occurence file
path  = "/Users/marcschubert/Documents/rnns/ErrorCategorization/ERROR_FILTERING.csv"
path = "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL_V3/ErrorCats/ERROR_FILTERING.csv"

data = pd.read_csv(path)
data.columns
os.chdir( "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL_V3/ErrorCats/")
### Functions######

def getErrorGroups(data, threshold):
    systematic = data[data["perc"]>=threshold]
    # below: no, that is wrong, because there are NAs
    #random = data[data["perc"]<threshold]
    random = data[~data["error_name"].isin(systematic["error_name"])]

    print(data.shape, systematic.shape, random.shape)
    out = {"systematic": list(systematic["error_name"]), 
    "random":list(random["error_name"])
    }
    return out

def writeJSON(data, filename):
    with open(filename, "w") as file:
        json.dump(data, file)

### Functions######

# filtering, only basic not fine grained errors
data = data[data["fineGrained"]=="basic"]

threshold = 0.1
names = getErrorGroups(data, threshold = threshold)
writeJSON(names, f"error_names_threshold{threshold}.json")

threshold = 0.05
names = getErrorGroups(data, threshold = threshold)
writeJSON(names, f"error_names_threshold{threshold}.json")

threshold = 0.01
names = getErrorGroups(data, threshold = threshold)
writeJSON(names, f"error_names_threshold{threshold}.json")







