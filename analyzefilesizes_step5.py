# analyze sizes of files step 5 .py
import os

# check how large the specific training sets wer.

directory = "/Users/marcschubert/Documents/rnns/Data/BeRNN_01/PreprocessedData_wResp_ALL"

for root, dirs, files in os.walk(directory):
    print(root, files)

