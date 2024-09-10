import os 
path = "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL"
os.chdir(path)


folders = os.listdir()
folders.remove(".DS_Store")
for folder in folders:
    files = os.listdir(folder)

    for file in files:
        if file.count("ERRORS"):
            print(file)
            os.remove(os.path.join(folder,file))

for folder in folders:
    files = os.listdir(folder)

    for file in files:
        if file.count("_ORIGINAL"):
            new_name = file.replace("_ORIGINAL", "")
            print(file, new_name)
            os.rename(os.path.join(folder,file), os.path.join(folder,new_name))


