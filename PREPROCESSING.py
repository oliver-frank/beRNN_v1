import numpy as np
import pandas as pd
import glob
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# todo: I think the main reason for the .xlsx problem are the .xslx files itself (produced by Gorilla),
#  change to other data form and retry

# Helper function to calculate the distance for circular activation ####################################################
def get_dist(original_dist):
    return np.minimum(abs(original_dist), 2 * np.pi - abs(original_dist))

# Helper function to calculate the circular activation
def add_x_loc(x_loc, pref):
    dist = get_dist(x_loc - pref)
    dist /= np.pi / 8
    return 0.8 * np.exp(-dist ** 2 / 2)

# Find the right questionare for the current session to extract the right meta information afterwards
def find_questionaire(target, list_allSessions, questionnare_files):
    for list in list_allSessions:
        for string in list:
            if string == target:
                for questionaire in questionnare_files:
                    if questionaire.split('-')[2] == list[0]:
                        return string, questionaire # , drugVector, sleepingQuality
    return None

# Function giving us the sleepingQuality and drugVector for the current session
def find_sleepingQuality_drugVector(opened_questionare, date):
    for index, row in opened_questionare.iterrows():
        if str(row['UTC Date and Time']).split(' ')[0] == date:
            sleepingQ = opened_questionare.at[index + 1, 'Question']
            sleepingR = opened_questionare.at[index + 1, 'Response']
            drugQ = opened_questionare.loc[index + 2:index + 18, 'Key'].to_list()
            drugR = opened_questionare.loc[index + 2:index + 18, 'Response'].to_list()
            # print(sleepingQ, sleepingR, drugQ, drugR)
            return sleepingQ, sleepingR, drugQ, drugR
    return None, None, None, None  # Return None if no match is found


########################################################################################################################
# todo: DM tasks #######################################################################################################
########################################################################################################################
# For debugging
# xlsxFile = 'W:\\AG_CSP\\Projekte\\BeRNN\\02_Daten\\BeRNN_main\\BeRNN_01\\1\\data_exp_149474-v2_task-6at8-9621849.xlsx'
def preprocess_DM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1], list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl', dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[
        ['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct',
         'Component Name', 'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
         'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
         'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
         'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
         'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
         'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
         'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
         'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
         'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
         'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
         'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
         'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
         'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
         'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
         'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
         'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
         'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti',
         'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'UTC Date and Time']]

    # Count and create batches
    incrementList = [i + 1 for i, name in enumerate(opened_xlsxFile_selection['Component Name']) if name == 'Fixation Timing']
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        numFixStepsTotal = 0
        numRespStepsTotal = 0

        # Calculate average fix, resp and total steps for this batch
        for j in incrementList[batchOn:batchOff]:
            # Accumulate step numbers
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            numFixSteps = round(currentTrial['Onset Time'][0] / 20)
            numRespSteps = round(currentTrial['Onset Time'][1] / 20)
            numFixStepsTotal += numFixSteps
            numRespStepsTotal += numRespSteps
            # Calculate average after cumulative addition of whole batch
            if j == incrementList[batchOff - 1]:
                numFixStepsAverage = int(numFixStepsTotal / batchLength)
                numRespStepsAverage = int(numRespStepsTotal / batchLength)
                totalStepsAverage = numFixStepsAverage + numRespStepsAverage
                # print(numFixStepsAverage,numRespStepsAverage,totalStepsAverage)

        finalSequenceList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)

        # Create final df for INPUT and OUPUT
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # Create Yang form
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 86))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 85]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        print('# DEBUGGING ###########################################################################################')
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
        print('# DEBUGGING ###########################################################################################')
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8,85], axis=2)
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,85], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]

        # INPUT ############################################################################################################
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # float all 000_000's on field units to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    if Input[i][j][k] == '000_000.png':
                        Input[i][j][k] = float(0)

        # Define modulation dictionaries for specific columns
        # mod1Dict = {'lowest': float(0.25), 'low': float(0.5), 'strong': float(0.75), 'strongest': float(1.0)}
        mod1Dict = {'lowest': float(0.5), 'low': float(1), 'strong': float(1.5), 'strongest': float(2)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 33):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod1Dict[Input[i][j][k].split('_')[0]]

        # Define modulation dictionaries for specific columns
        # mod2Dict = {'right.png': float(0.25), 'down.png': float(0.5), 'left.png': float(0.75), 'up.png': float(1.0)}
        mod2Dict = {'right.png': float(0.5), 'down.png': float(1), 'left.png': float(1.5), 'up.png': float(2)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(33, 65):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod2Dict[Input[i][j][k].split('_')[1]]

        # float all field values of fixation period to 0
        for i in range(0, numFixStepsAverage):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    Input[i][j][k] = float(0)

        # Add input gradient activation
        # Create default hyperparameters for network
        num_ring, n_eachring, n_rule = 2, 32, 12
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                currentTimeStepModOne = Input[i][j][1:33]
                currentTimeStepModTwo = Input[i][j][33:65]
                # Allocate first unit ring
                unitRingMod1 = np.zeros(32, dtype='float32')
                unitRingMod2 = np.zeros(32, dtype='float32')

                # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
                NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
                NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
                if len(NonZero_Mod1) != 0:
                    # Accumulating all activities for both unit rings together
                    for k in range(0, len(NonZero_Mod1)):
                        currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                        currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                        currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                        currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                        # add one gradual activated stim to final form
                        currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                        currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                        # Add all activations for one trial together
                        unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                        unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)

                    # Store
                    currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                    Input[i][j][0:78] = currentFinalRow

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(0, Input.shape[2]):
                    Input[i][j][k] = np.float32(Input[i][j][k])
        # Also change dtype for entire array
        Input = Input.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = participant+'-'+'month_'+str(month)+'-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                         xlsxFile.split('_')[3].split('-')[0]+'_'+ xlsxFile.split('_')[3].split('-')[1]+'-'+'Input'
        np.save(input_filename, Input)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT ###########################################################################################################
        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0.05)
        # float all field units of response epoch to 0
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0)
        # float all fixation outputs during response period to 0.05
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                Output[i][j][1] = float(0.05)

        # Define an output dictionary for specific response values
        outputDict = {'U': 32, 'R': 8, 'L': 24, 'D': 16}

        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                if Output[i][j][0] != 'noResponse' and Output[i][j][0] != 'NoResponse':
                    Output[i][j][outputDict[Output[i][j][0]]] = float(0.85)
                else:
                    for k in range(2, 34):
                        Output[i][j][k] = float(0.05)

        # Drop unnecessary first column with response information
        Output = np.delete(Output, [0], axis=2)
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        # Add output gradient activation
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                currentTimeStepOutput = Output[i][j][1:33]
                # Allocate first unit ring
                unitRingOutput = np.zeros(32, dtype='float32')
                # Get non-zero values of time steps
                nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
                # Float first fixations rows with -1
                for k in range(0, numFixStepsAverage):
                    y_loc[k][j] = np.float(-1)

                if len(nonZerosOutput) == 1:
                    # Get activity and model gradient activation around it
                    currentOutputLoc = pref[nonZerosOutput[0]]
                    currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
                    unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                    # Store
                    currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                    Output[i][j][0:33] = currentFinalRow
                    # Complete y_loc matrix
                    for k in range(numFixStepsAverage, totalStepsAverage):
                        y_loc[k][j] = pref[nonZerosOutput[0]]

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                for k in range(0, Output.shape[2]):
                    Output[i][j][k] = np.float32(Output[i][j][k])
        # Also change dtype for entire array
        Output = Output.astype('float32')

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Output'
        np.save(output_filename, Output)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# todo: EF tasks #######################################################################################################
########################################################################################################################
# For debugging
# xlsxFile = 'W:\\AG_CSP\\Projekte\\BeRNN\\02_Daten\\BeRNN_main\\BeRNN_01\\1\\data_exp_149474-v2_task-2p6f-9621849.xlsx'
def preprocess_EF(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[
                      ['Spreadsheet', 'TimeLimit', 'Onset Time', 'Response', 'Spreadsheet: CorrectAnswer', 'Correct',
                       'Component Name', 'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
                       'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
                       'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
                       'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
                       'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
                       'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
                       'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2',
                       'Spreadsheet: Field 3', 'Spreadsheet: Field 4', 'Spreadsheet: Field 5', 'Spreadsheet: Field 6',
                       'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', 'Spreadsheet: Field 10',
                       'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18',
                       'Spreadsheet: Field 19', 'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22',
                       'Spreadsheet: Field 23', 'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26',
                       'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29', 'Spreadsheet: Field 30',
                       'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM', 'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti',
                       'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'UTC Date and Time']]

    # Define batch size and create batches
    incrementList = [i + 1 for i, name in enumerate(opened_xlsxFile_selection['Component Name']) if name == 'Fixation Timing']
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        numFixStepsTotal = 0
        numRespStepsTotal = 0

        # Calculate average fix, resp and total steps for this batch
        for j in incrementList[batchOn:batchOff]:
            # Accumulate step numbers
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            numFixSteps = round(currentTrial['Onset Time'][0] / 20)
            numRespSteps = round(currentTrial['Onset Time'][1] / 20)
            numFixStepsTotal += numFixSteps
            numRespStepsTotal += numRespSteps
            # Calculate average after cumulative addition of whole batch
            if j == incrementList[batchOff - 1]:
                numFixStepsAverage = int(numFixStepsTotal / batchLength)
                numRespStepsAverage = int(numRespStepsTotal / batchLength)
                totalStepsAverage = numFixStepsAverage + numRespStepsAverage
                # print(numFixStepsAverage,numRespStepsAverage,totalStepsAverage)

        finalSequenceList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)

        # Create final df for INPUT and OUPUT
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # Create Yang form
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 86))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 85]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        print('# DEBUGGING ###########################################################################################')
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
        print('# DEBUGGING ###########################################################################################')
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8,85], axis=2)
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,85], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]

        # INPUT ############################################################################################################
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # float all 000_000's on field units to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    if Input[i][j][k] == '000_000.png':
                        Input[i][j][k] = float(0)

        # Define modulation dictionaries for specific columns
        # mod1Dict = {'green': float(0.5), 'red': float(1.0)}
        mod1Dict = {'green': float(1.0), 'red': float(2.0)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 33):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod1Dict[Input[i][j][k].split('_')[0]]

        # Define modulation dictionaries for specific columns
        # mod2Dict = {'right.png': float(0.2), 'down.png': float(0.4), 'left.png': float(0.6), 'up.png': float(0.8), 'X.png': float(1.0)}
        mod2Dict = {'right.png': float(0.4), 'down.png': float(0.8), 'left.png': float(1.2), 'up.png': float(1.6), 'X.png': float(2.0)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(33, 65):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod2Dict[Input[i][j][k].split('_')[1]]

        # float all field values of fixation period to 0
        for i in range(0, numFixStepsAverage):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    Input[i][j][k] = float(0)

        # Add input gradient activation
        # Create default hyperparameters for network
        num_ring, n_eachring, n_rule = 2, 32, 12
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                currentTimeStepModOne = Input[i][j][1:33]
                currentTimeStepModTwo = Input[i][j][33:65]
                # Allocate first unit ring
                unitRingMod1 = np.zeros(32, dtype='float32')
                unitRingMod2 = np.zeros(32, dtype='float32')

                # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
                NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
                NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
                if len(NonZero_Mod1) != 0:
                    # Accumulating all activities for both unit rings together
                    for k in range(0, len(NonZero_Mod1)):
                        currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                        currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                        currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                        currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                        # add one gradual activated stim to final form
                        currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                        currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                        # Add all activations for one trial together
                        unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                        unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)

                    # Store
                    currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                    Input[i][j][0:78] = currentFinalRow

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(0, Input.shape[2]):
                    Input[i][j][k] = np.float32(Input[i][j][k])
        # Also change dtype for entire array
        Input = Input.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = (participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Input')
        np.save(input_filename, Input)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT ###########################################################################################################
        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0.05)
        # float all field units of response epoch to 0
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0)
        # float all fixation outputs during response period to 0.05
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                Output[i][j][1] = float(0.05)

        # Define an output dictionary for specific response values
        outputDict = {'U': 32, 'R': 8, 'L': 24, 'D': 16}

        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                if Output[i][j][0] != 'noResponse' and Output[i][j][0] != 'NoResponse':
                    Output[i][j][outputDict[Output[i][j][0]]] = float(0.85)
                else:
                    for k in range(2, 34):
                        Output[i][j][k] = float(0.05)

        # Drop unnecessary first column
        Output = np.delete(Output, [0], axis=2)
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        # Add output gradient activation
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                currentTimeStepOutput = Output[i][j][1:33]
                # Allocate first unit ring
                unitRingOutput = np.zeros(32, dtype='float32')
                # Get non-zero values of time steps
                nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
                # Float first fixations rows with -1
                for k in range(0, numFixStepsAverage):
                    y_loc[k][j] = np.float(-1)

                if len(nonZerosOutput) == 1:
                    # Get activity and model gradient activation around it
                    currentOutputLoc = pref[nonZerosOutput[0]]
                    currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
                    unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                    # Store
                    currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                    Output[i][j][0:33] = currentFinalRow
                    # Complete y_loc matrix
                    for k in range(numFixStepsAverage, totalStepsAverage):
                        y_loc[k][j] = pref[nonZerosOutput[0]]

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                for k in range(0, Output.shape[2]):
                    Output[i][j][k] = np.float32(Output[i][j][k])
        # Also change dtype for entire array
        Output = Output.astype('float32')

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Output'
        np.save(output_filename, Output)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# todo: RP tasks #######################################################################################################
########################################################################################################################
# For debugging
# xlsxFile = 'W:\\AG_CSP\\Projekte\\BeRNN\\02_Daten\\BeRNN_main\\BeRNN_01\\1\\data_exp_149474-v2_task-bert-9621849.xlsx'
def preprocess_RP(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Spreadsheet: CorrectAnswer1', 'Correct', 'Component Name', \
                           'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                           'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                           'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                           'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                           'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                           'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                           'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                           'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                           'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                           'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                           'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                           'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                           'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                           'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                           'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display', 'UTC Date and Time']]

    # Define batch size and create batches
    incrementList = [i + 1 for i, name in enumerate(opened_xlsxFile_selection['Component Name']) if name == 'Fixation Timing']
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        numFixStepsTotal = 0
        numRespStepsTotal = 0

        # Calculate average fix, resp and total steps for this batch
        for j in incrementList[batchOn:batchOff]:
            # Accumulate step numbers
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            numFixSteps = round(currentTrial['Onset Time'][0] / 20)
            numRespSteps = round(currentTrial['Onset Time'][1] / 20)
            numFixStepsTotal += numFixSteps
            numRespStepsTotal += numRespSteps
            # Calculate average after cumulative addition of whole batch
            if j == incrementList[batchOff - 1]:
                numFixStepsAverage = int(numFixStepsTotal / batchLength)
                numRespStepsAverage = int(numRespStepsTotal / batchLength)
                totalStepsAverage = numFixStepsAverage + numRespStepsAverage
                # print(numFixStepsAverage,numRespStepsAverage,totalStepsAverage)

        finalSequenceList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)

        # Create final df for INPUT and OUPUT
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # Create Yang form
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 87))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0,0,86]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        print('# DEBUGGING ###########################################################################################')
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
        print('# DEBUGGING ###########################################################################################')
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0,0,0],
                     'timeLimit': finalTrialsList_array[0,0,1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8, 85,86], axis=2) # todo: 77 statt 85 ????
        Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,86], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]

        # INPUT ############################################################################################################
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # float all 000_000's on field units to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    if Input[i][j][k] == '000_000.png':
                        Input[i][j][k] = float(0)

        # Define modulation dictionaries for specific columns
        # mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
        #             'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}
        mod1Dict = {'red': 0.16, 'rust': 0.32, 'orange': 0.5, 'amber': 0.66, 'yellow': 0.84, 'lime': 1.0,'green': 1.16, \
                    'moss': 1.32, 'blue': 1.5, 'violet': 1.66, 'magenta': 1.84, 'purple': 2.0}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 33):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod1Dict[Input[i][j][k].split('_')[0]]

        # Define modulation dictionaries for specific columns
        mod2Dict = {'triangle.png': float(0.4), 'pentagon.png': float(0.8), 'heptagon.png': float(1.2), 'nonagon.png': float(1.6), 'circle.png': float(2.0)}
        # mod2Dict = {'triangle.png': float(0.2), 'pentagon.png': float(0.4), 'heptagon.png': float(0.6), 'nonagon.png': float(0.8), 'circle.png': float(1.0)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(33, 65):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod2Dict[Input[i][j][k].split('_')[1]]

        # float all field values of fixation period to 0
        for i in range(0, numFixStepsAverage):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    Input[i][j][k] = float(0)

        # Add input gradient activation
        # Create default hyperparameters for network
        num_ring, n_eachring, n_rule = 2, 32, 12
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                currentTimeStepModOne = Input[i][j][1:33]
                currentTimeStepModTwo = Input[i][j][33:65]
                # Allocate first unit ring
                unitRingMod1 = np.zeros(32, dtype='float32')
                unitRingMod2 = np.zeros(32, dtype='float32')

                # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
                NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
                NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
                if len(NonZero_Mod1) != 0:
                    # Accumulating all activities for both unit rings together
                    for k in range(0, len(NonZero_Mod1)):
                        currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                        currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                        currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                        currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                        # add one gradual activated stim to final form
                        currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                        currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                        # Add all activations for one trial together
                        unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                        unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
                    # Store
                    currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                    Input[i][j][0:78] = currentFinalRow

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(0, Input.shape[2]):
                    Input[i][j][k] = np.float32(Input[i][j][k])
        # Also change dtype for entire array
        Input = Input.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = (participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Input')
        np.save(input_filename, Input)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT ###########################################################################################################
        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0.05)
        # float all field units of response epoch to 0
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(2, 34):
                    Output[i][j][k] = float(0)
        # float all fixation outputs during response period to 0.05
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                Output[i][j][1] = float(0.05)

        # Assign field units to their according participant response value after fixation period
        outputDict_RP_1 = {'Image 2': 2, 'Image 4': 4, 'Image 6': 6, 'Image 8': 8, 'Image 10': 10, 'Image 12': 12, 'Image 14': 14,\
            'Image 16': 16, 'Image 18': 18, 'Image 20': 20, 'Image 22': 22, 'Image 24': 24, 'Image 26': 26, 'Image 28': 28, 'Image 30': 30, 'Image 32': 32}

        outputDict_RP_2 = {'Image 1': 1, 'Image 3': 3, 'Image 5': 5, 'Image 7': 7, 'Image 9': 9, 'Image 11': 11, 'Image 13': 13, 'Image 15': 15,\
            'Image 17': 17, 'Image 19': 19, 'Image 21': 21, 'Image 23': 23, 'Image 25': 25, 'Image 27': 27, 'Image 29': 29, 'Image 31': 31}

        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                # Get the right dictionary
                if Output[i][j][34].split(' RP')[0] == 'Display 1':
                    outputDict = outputDict_RP_1
                elif Output[i][j][34].split(' RP')[0] == 'Display 2':
                    outputDict = outputDict_RP_2

                if Output[i][j][0] != 'screen' and Output[i][j][0] != 'noResponse' and Output[i][j][0] != 'NoResponse':
                    Output[i][j][outputDict[Output[i][j][0]]] = np.float32(0.85)
                else:
                    for k in range(2, 34):
                        Output[i][j][k] = np.float32(0.05)

        # Drop unnecessary columns
        Output = np.delete(Output, [0,34], axis=2)
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        # Add output gradient activation
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                currentTimeStepOutput = Output[i][j][1:33]
                # Allocate first unit ring
                unitRingOutput = np.zeros(32, dtype='float32')
                # Get non-zero values of time steps
                nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
                # Float first fixations rows with -1
                for k in range(0, numFixStepsAverage):
                    y_loc[k][j] = np.float(-1)

                if len(nonZerosOutput) == 1:
                    # Get activity and model gradient activation around it
                    currentOutputLoc = pref[nonZerosOutput[0]]
                    currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
                    unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                    # Store
                    currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                    Output[i][j][0:33] = currentFinalRow
                    # Complete y_loc matrix
                    for k in range(numFixStepsAverage, totalStepsAverage):
                        y_loc[k][j] = pref[nonZerosOutput[0]]

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                for k in range(0, Output.shape[2]):
                    Output[i][j][k] = np.float32(Output[i][j][k])
        # Also change dtype for entire array
        Output = Output.astype('float32')

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Output'
        np.save(output_filename, Output)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

########################################################################################################################
# todo: WM tasks #######################################################################################################
########################################################################################################################
# For debugging
# batchfile_location = 'W:\\AG_CSP\\Projekte\\BeRNN\\02_Daten\\BeRNN_main\\BeRNN_01\\1\\data_exp_149474-v2_task-fhgh-9621849.xlsx'
def preprocess_WM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength):
    # Initialize columns with default values
    opened_xlsxFile.loc[:, 'Fixation input'] = 1
    opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
    opened_xlsxFile.loc[:, 'DM'] = 0
    opened_xlsxFile.loc[:, 'DM Anti'] = 0
    opened_xlsxFile.loc[:, 'EF'] = 0
    opened_xlsxFile.loc[:, 'EF Anti'] = 0
    opened_xlsxFile.loc[:, 'RP'] = 0
    opened_xlsxFile.loc[:, 'RP Anti'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
    opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
    opened_xlsxFile.loc[:, 'WM'] = 0
    opened_xlsxFile.loc[:, 'WM Anti'] = 0
    opened_xlsxFile.loc[:, 'WM Ctx1'] = 0

    # Find questionare associated to current .xlsx file
    taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
    opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})

    # Select specific columns from the DataFrame
    opened_xlsxFile_selection = opened_xlsxFile[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Object ID', 'Spreadsheet: CorrectAnswer', 'Correct', 'Component Name', \
                       'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
                       'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
                       'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
                       'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
                       'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
                       'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
                       'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
                       'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
                       'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
                       'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
                       'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display', 'UTC Date and Time']]

    # Define batch size and create batches
    incrementList = [i + 1 for i, name in enumerate(opened_xlsxFile_selection['Component Name']) if name == 'Fixation Timing']
    numberBatches = len(incrementList) // batchLength

    # Split the data into batches based on the fixation timing component
    for batchNumber in range(numberBatches):
        batchOn = batchNumber * batchLength
        batchOff = batchNumber * batchLength + batchLength
        numFixStepsTotal = 0
        numRespStepsTotal = 0

        # Calculate average fix, resp and total steps for this batch
        for j in incrementList[batchOn:batchOff]:
            # Accumulate step numbers
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            numFixSteps = round(currentTrial['Onset Time'][0] / 20)
            numRespSteps = round(currentTrial['Onset Time'][1] / 20)
            numFixStepsTotal += numFixSteps
            numRespStepsTotal += numRespSteps
            if j == incrementList[batchOff - 1]:
                numFixStepsAverage = int(numFixStepsTotal / batchLength)
                numRespStepsAverage = int(numRespStepsTotal / batchLength)
                totalStepsAverage = numFixStepsAverage + numRespStepsAverage
                # print(numFixStepsAverage,numRespStepsAverage,totalStepsAverage)

        finalSequenceList = []
        # Create sequences of every trial in the current batch with the previous calculated time steps
        for j in incrementList[batchOn:batchOff]:
            currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
            currentSequenceList = []
            for k in range(totalStepsAverage):
                sequence = [currentTrial.iloc[0]]
                currentSequenceList.append(sequence)
            finalSequenceList.append(currentSequenceList)

        # Create final df for INPUT and OUPUT
        newOrderSequenceList = []
        # Append all the time steps accordingly to a list
        for j in range(0, totalStepsAverage):
            for i in range(0, len(finalSequenceList)):
                newOrderSequenceList.append(finalSequenceList[i][j])

        # Create Yang form
        finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 88))
        # Create meta dict before deleting necessary information from current trials List
        date = str(finalTrialsList_array[0, 0, 87]).split(' ')[0]
        sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
        print('# DEBUGGING ###########################################################################################')
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
        print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
        print('# DEBUGGING ###########################################################################################')
        # Catch meta data now that you have all the necessary
        meta_dict = {'date_time': str(finalTrialsList_array[0, 0, 85]), 'difficultyLevel': finalTrialsList_array[0, 0, 0],
                     'timeLimit': finalTrialsList_array[0, 0, 1], 'sleepingQualityQuestion': sleepingQualityQuestion,
                     'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
                     'drugVectorValue': drugVectorValue}
        # Create one input file and one output file
        Input, Output = finalTrialsList_array, finalTrialsList_array
        Input = np.delete(Input, [0,1,2,3,4,5,6,7,8,86,87], axis=2)
        Output = np.delete(Output, np.s_[0,1,2,5,6,7,8,87], axis=2)
        Output = np.delete(Output, np.s_[34:78], axis=2)
        Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]

        # INPUT ############################################################################################################
        # float all fixation input values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                Input[i][j][0] = float(1)
        # float all task values to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(65, Input.shape[2]):
                    Input[i][j][k] = float(0)

        # Define a task dictionary for specific task-related columns
        taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
                    'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}

        # float all task values to 1
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
                else:
                    Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
                    taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]

        # float all 000_000's on field units to 0
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    if Input[i][j][k] == '000_000.png':
                        Input[i][j][k] = float(0)

        # Define modulation dictionaries for specific columns
        mod1Dict = {'red': 0.16, 'rust': 0.34, 'orange': 0.5, 'amber': 0.66, 'yellow': 0.84, 'lime': 1.0, 'green': 1.16, \
                    'moss': 1.34, 'blue': 1.5, 'violet': 1.66, 'magenta': 1.84, 'purple': 2.0}
        # mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
        #             'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(1, 33):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod1Dict[Input[i][j][k].split('_')[0]]

        # Define modulation dictionaries for specific columns
        # mod2Dict = {'triangle.png': float(0.2), 'pentagon.png': float(0.4), 'heptagon.png': float(0.6), 'nonagon.png': float(0.8), 'circle.png': float(1.0)}
        mod2Dict = {'triangle.png': float(0.4), 'pentagon.png': float(0.8), 'heptagon.png': float(1.2), 'nonagon.png': float(1.6), 'circle.png': float(2.0)}

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(33, 65):
                    if Input[i][j][k] != 0:
                        Input[i][j][k] = mod2Dict[Input[i][j][k].split('_')[1]]

        # float all field values of fixation period to 0
        for i in range(0, numFixStepsAverage):
            for j in range(0, Input.shape[1]):
                for k in range(1, 65):
                    Input[i][j][k] = float(0)

        # Add input gradient activation
        # Create default hyperparameters for network
        num_ring, n_eachring, n_rule = 2, 32, 12
        n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
        pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)

        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                currentTimeStepModOne = Input[i][j][1:33]
                currentTimeStepModTwo = Input[i][j][33:65]
                # Allocate first unit ring
                unitRingMod1 = np.zeros(32, dtype='float32')
                unitRingMod2 = np.zeros(32, dtype='float32')

                # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
                NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
                NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
                if len(NonZero_Mod1) != 0:
                    # Accumulating all activities for both unit rings together
                    for k in range(0, len(NonZero_Mod1)):
                        currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
                        currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
                        currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
                        currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
                        # add one gradual activated stim to final form
                        currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
                        currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
                        # Add all activations for one trial together
                        unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
                        unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)

                    # Store
                    currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
                    Input[i][j][0:78] = currentFinalRow

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Input.shape[0]):
            for j in range(0, Input.shape[1]):
                for k in range(0, Input.shape[2]):
                    Input[i][j][k] = np.float32(Input[i][j][k])
        # Also change dtype for entire array
        Input = Input.astype('float32')

        # Save input data
        os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
        input_filename = (participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                    xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Input')
        np.save(input_filename, Input)

        # Sanity check
        print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])

        # OUTPUT ###########################################################################################################
        # float all field units during fixation epoch on 0.05
        for i in range(0, numFixStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(3, 35):
                    Output[i][j][k] = float(0.05)
        # float all field units of response epoch to 0
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                for k in range(3, 35):
                    Output[i][j][k] = float(0)
        # float all fixation outputs during response period to 0.05
        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                Output[i][j][2] = float(0.05)

        # Assign field units to their according participant response value after fixation period
        outputDict_WM = {'Image 1': 1, 'Image 2': 2, 'Image 3': 3, 'Image 4': 4, 'Image 5': 5, 'Image 6': 6, 'Image 7': 7,\
            'Image 8': 8, 'Image 9': 9, 'Image 10': 10, 'Image 11': 11, 'Image 12': 12, 'Image 13': 13, 'Image 14': 14,\
            'Image 15': 15, 'Image 16': 16, 'Image 17': 17, 'Image 18': 18, 'Image 19': 19, 'Image 20': 20, 'Image 21': 21,\
            'Image 22': 22, 'Image 23': 23, 'Image 24': 24, 'Image 25': 25, 'Image 26': 26, 'Image 27': 27, 'Image 28': 28,\
            'Image 29': 29, 'Image 30': 30, 'Image 31': 31, 'Image 32': 32}

        outputDict_WM_Ctx = {'object-1591': 8, 'object-1593': 8, 'object-1595': 8, 'object-1597': 8, 'object-1592': 24,\
            'object-1594': 24, 'object-1596': 24, 'object-1598': 24}

        for i in range(numFixStepsAverage, totalStepsAverage):
            for j in range(0, Output.shape[1]):
                if isinstance(Output[i][j][35], str):
                    # Get the right dictionary
                    if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4 or opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1] == 'Anti':
                        outputDict = outputDict_WM
                        chosenColumn = 0
                    else:
                        outputDict = outputDict_WM_Ctx
                        chosenColumn = 1

                    if Output[i][j][0] != 'screen' and Output[i][j][0] != 'noResponse' and Output[i][j][0] != 'NoResponse'\
                            and Output[i][j][1] != 'Fixation Cross' and Output[i][j][1] != 'Response':
                        Output[i][j][outputDict[Output[i][j][chosenColumn]]] = np.float32(0.85)
                    else:
                        for k in range(3, 35):
                            Output[i][j][k] = np.float32(0.05)
                else:
                    for k in range(3, 35):
                        Output[i][j][k] = np.float32(0.05)

        # Drop unnecessary columns
        Output = np.delete(Output, [0,1,35], axis=2)
        # Pre-allocate y-loc matrix; needed for later validation
        y_loc = np.zeros((Output.shape[0], Output.shape[1]))

        # Add output gradient activation
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                currentTimeStepOutput = Output[i][j][1:33]
                # Allocate first unit ring
                unitRingOutput = np.zeros(32, dtype='float32')
                # Get non-zero values of time steps
                nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
                # Float first fixations rows with -1
                for k in range(0, numFixStepsAverage):
                    y_loc[k][j] = np.float(-1)

                if len(nonZerosOutput) == 1:
                    # Get activity and model gradient activation around it
                    currentOutputLoc = pref[nonZerosOutput[0]]
                    currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
                    unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
                    # Store
                    currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
                    Output[i][j][0:33] = currentFinalRow
                    # Complete y_loc matrix
                    for k in range(numFixStepsAverage, totalStepsAverage):
                        y_loc[k][j] = pref[nonZerosOutput[0]]

        # Change dtype of every element in matrix to float32 for later validation functions
        for i in range(0, Output.shape[0]):
            for j in range(0, Output.shape[1]):
                for k in range(0, Output.shape[2]):
                    Output[i][j][k] = np.float32(Output[i][j][k])
        # Also change dtype for entire array
        Output = Output.astype('float32')

        # Save output data
        output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                          xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Output'
        np.save(output_filename, Output)
        # Save y_loc data
        yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
        np.save(yLoc_filename, y_loc)
        # Save meta data
        meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
                        xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
        with open('{}.json'.format(meta_filename), 'w') as json_file:
            json.dump(meta_dict, json_file)

        # Sanity check
        print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
        print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])


# todo: ################################################################################################################
# todo: Preprocessing ##################################################################################################
# todo: ################################################################################################################
def check_permissions(file_path):
    permissions = {
        'read': os.access(file_path, os.R_OK),
        'write': os.access(file_path, os.W_OK),
        'execute': os.access(file_path, os.X_OK)
    }
    return permissions

# Create right path - os.getcwd() should be set to PycharmProject mulitask_BeRNN
dataFolder = "Data"
participant = 'BeRNN_03'
main_folder = 'PreprocessedData_encodingX2'
main_path = os.path.join(os.getcwd(),dataFolder, participant, main_folder)
# Create Folder Structure
subfolders = ['DM', 'DM_Anti', 'EF', 'EF_Anti', 'RP', 'RP_Anti', 'RP_Ctx1', 'RP_Ctx2',
              'WM', 'WM_Anti', 'WM_Ctx1', 'WM_Ctx2']

if not os.path.exists(main_path):
    os.makedirs(main_path)

for folder in subfolders:
    path = os.path.join(main_path, folder)
    if not os.path.exists(path):
        os.makedirs(path)

# Processing path allocation
processing_path = os.path.join(os.getcwd(),dataFolder, participant)
# Months
months = ['1','2','3'] # todo: add here months
# list of task names in each session; very first name is the one of the associated questionare
list_d4fh = ['d4fh', 'b4ya', 'bert', '7py6', 'bx2n', '2p6f', '9ivx', 'fhgh', 'k1jg', '6x4x', 'aiut', 'ar8p', '627g']
list_2353 = ['2353', '6vnh', '113d', 'pbq6', 'q4ei', 'u7od', '9qcv', '4lrw', 'u31n', 'jr36', 'hia1', 'odic', 'qkw4']
list_sdov = ['sdov', 'ohf2', '7y8y', 'p3op', '715w', 'hbck', '8dc4', 'pfww', 'kid1', 'z84v', 'qfff', 'o9l4', 'fiv9']
list_h3ph = ['h3ph', 'kvmz', 'gpb4', 'x1mk', 'qxae', '4cnx', '9wpb', 'ujcn', 'o3t1', 'qf8s', 't271', 'lypz', '7l94']
list_jwd5 = ['jwd5', 'zner', 'fvox', 'qqvk', 'qilu', 'xqal', 'q9wz', 'p7mk', '2kln', 'ifgy', '6at8', 'zolj', '9utw']
list_allSessions = [list_d4fh, list_2353, list_sdov, list_h3ph, list_jwd5]
# Go through all .xlsx files in the defined months for one participant and save them in PreprocessedData folder's subfolders
for month in months:
    processing_path_month = os.path.join(processing_path, month)
    if os.path.exists(processing_path_month):
        pattern = os.path.join(processing_path_month, '*.xlsx')
        xlsx_files = glob.glob(pattern)
        task_files = [os.path.basename(file) for file in xlsx_files if 'questionnaire' not in os.path.basename(file).lower()]
        questionnare_files = [os.path.basename(file) for file in xlsx_files if 'task' not in os.path.basename(file).lower()]
        # Iterate through all .xlsx files in current month folder
        for xlsxFile in task_files:
            file_path = os.path.join(processing_path_month, xlsxFile)
            print(f"Processing file: {file_path}")
            permissions = check_permissions(file_path)

            print(f"Read: {'Yes' if permissions['read'] else 'No'}")
            print(f"Write: {'Yes' if permissions['write'] else 'No'}")
            print(f"Execute: {'Yes' if permissions['execute'] else 'No'}")
            if permissions['read']:
                if os.path.isfile(file_path):
                    try:
                        opened_xlsxFile = pd.read_excel(file_path, engine='openpyxl')
                        print(file_path, ' successfully opened')
                        sequence_on, sequence_off, batchLength = 0, 40, 40
                        try:
                            # Preprocess the xlsxFile according to its task type and directly save it to the right directory
                            if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'DM':
                                preprocess_DM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                            elif opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'EF':
                                preprocess_EF(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                            elif opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'RP':
                                preprocess_RP(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                            if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'WM':
                                preprocess_WM(opened_xlsxFile, questionnare_files, list_allSessions, sequence_on, sequence_off, batchLength)
                        except Exception as e:
                            print(f"An error occurred with file {xlsxFile}: {e}")
                    except Exception as e:
                        print(f"An error occurred with file {xlsxFile}: {e}")
                else:
                    print(f"File not found: {file_path}")
            else:
                print(f"Read permission denied for file: {file_path}")
    else:
        print(f"Month directory not found: {processing_path_month}")



# todo: ################################################################################################################
# todo: DEBUG ZONE #####################################################################################################
# todo: ################################################################################################################

# Input = np.load(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\multitask_BeRNN\\Data\\BeRNN_01\\PreprocessedData\\RP_Ctx1',\
#                                    'BeRNN_01-month_1-batch_0-RP_Ctx1-task_113d-Input.npy'))
# Output = np.load(os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\multitask_BeRNN\\Data\\BeRNN_01\\PreprocessedData\\RP_Ctx1',\
#                                     'BeRNN_01-month_1-batch_0-RP_Ctx1-task_113d-Output.npy'))
#
# filename = os.path.join('Z:\\Desktop\\ZI\\PycharmProjects\\multitask_BeRNN\\Data\\BeRNN_01\\PreprocessedData\\RP_Ctx1',\
#                                    'BeRNN_01-month_1-batch_0-RP_Ctx1-task_113d-Meta.json')
# with open(filename, 'r') as json_file:
#     dict = json.load(json_file)
#
# # Visualiza Input and Output
# plt.imshow(Input[25], cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()
# plt.imshow(Output[25], cmap='hot', interpolation='nearest')
# plt.colorbar()
# plt.show()



# todo: Super weird error with 'None' xlsx files, when I iterate oer them here, there are no errors. When I do that above
# todo: there are some errors, when I do it on linux, every file is seen as a None and cannot be iterated
# xlsxFile = 'data_exp_152443-v2_task-113d-9761627.xlsx'
# month = '1'
# processing_path_month = os.path.join(processing_path, month)
# file_path = os.path.join(processing_path_month, xlsxFile)
#
# pd.read_excel(file_path, engine='openpyxl')
#
# xlsxFileList = [f for f in os.listdir(processing_path_month) if f.endswith('.xlsx')]
# # Iterate through all .xlsx files in current month folder
# for xlsxFile in xlsxFileList:
#     print(xlsxFile)
#     file_path = os.path.join(processing_path_month, xlsxFile)
#     try:
#         opened_xlsxFile = pd.read_excel(file_path, engine='openpyxl')
#         print('File opened')
#     except Exception as e:
#         print(f"An error occurred with file {xlsxFile}: {e}")

# file_path = 'Z:/Desktop/ZI/PycharmProjects/multitask_BeRNN/Data/BeRNN_04/1/data_exp_152443-v2_task-113d-9761627.xlsx'
# opened_xlsxFile = pd.read_excel(file_path, engine='openpyxl')
#
# if opened_xlsxFile['Spreadsheet'][0].split('_')[0] == 'RP':
#     preprocess_RP(opened_xlsxFile, xlsxFileList, list_allSessions, sequence_on, sequence_off, batchLength)
#
# # Initialize columns with default values
# opened_xlsxFile.loc[:, 'Fixation input'] = 1
# opened_xlsxFile.loc[:, 'Fixation output'] = 0.8
# opened_xlsxFile.loc[:, 'DM'] = 0
# opened_xlsxFile.loc[:, 'DM Anti'] = 0
# opened_xlsxFile.loc[:, 'EF'] = 0
# opened_xlsxFile.loc[:, 'EF Anti'] = 0
# opened_xlsxFile.loc[:, 'RP'] = 0
# opened_xlsxFile.loc[:, 'RP Anti'] = 0
# opened_xlsxFile.loc[:, 'RP Ctx1'] = 0
# opened_xlsxFile.loc[:, 'RP Ctx2'] = 0
# opened_xlsxFile.loc[:, 'WM'] = 0
# opened_xlsxFile.loc[:, 'WM Anti'] = 0
# opened_xlsxFile.loc[:, 'WM Ctx1'] = 0
#
#
# # def find_questionaire(target, list_allSessions, xlsxFileList):
# target = opened_xlsxFile['Tree Node Key'][0].split('-')[1]
# try:
#     for list in list_allSessions:
#         # print(list)
#         for string in list:
#             # print(string)
#             if string == target:
#                 for questionaire in questionnare_files:
#                     print(questionaire)
#                     if questionaire.split('-')[2] == list[0]:
#                         print('FOUND')
#
#
#                     # return string, questionaire # , drugVector, sleepingQuality
#     # return None
#
# import glob
#
# pattern = os.path.join(processing_path_month, '*.xlsx')
# xlsx_files = glob.glob(pattern)
# task_files = [file for file in xlsx_files if 'questionnaire' not in os.path.basename(file).lower()]
# questionnare_files = [file for file in xlsx_files if 'task' not in os.path.basename(file).lower()]
#
# # Find questionare associated to current .xlsx file
# taskString, final_questionaire = find_questionaire(opened_xlsxFile['Tree Node Key'][0].split('-')[1],list_allSessions, questionnare_files)
#
#
#
# opened_questionare = pd.read_excel(os.path.join(processing_path_month, final_questionaire), engine='openpyxl',dtype={'column_name': 'float64'})
#
# # Select specific columns from the DataFrame
# opened_xlsxFile_selection = opened_xlsxFile[['Spreadsheet', 'TimeLimit', 'Onset Time', 'Object Name', 'Spreadsheet: CorrectAnswer1', 'Correct', 'Component Name', \
#                        'Fixation input', 'Fixation output', 'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4',\
#                        'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9',\
#                        'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
#                        'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
#                        'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23', 'Spreadsheet: Field 24',\
#                        'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28', 'Spreadsheet: Field 29',\
#                        'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', \
#                        'Spreadsheet: Field 1', 'Spreadsheet: Field 2', 'Spreadsheet: Field 3', 'Spreadsheet: Field 4', \
#                        'Spreadsheet: Field 5', 'Spreadsheet: Field 6', 'Spreadsheet: Field 7', 'Spreadsheet: Field 8', 'Spreadsheet: Field 9', \
#                        'Spreadsheet: Field 10', 'Spreadsheet: Field 11', 'Spreadsheet: Field 12', 'Spreadsheet: Field 13', 'Spreadsheet: Field 14',\
#                        'Spreadsheet: Field 15', 'Spreadsheet: Field 16', 'Spreadsheet: Field 17', 'Spreadsheet: Field 18', 'Spreadsheet: Field 19',\
#                        'Spreadsheet: Field 20', 'Spreadsheet: Field 21', 'Spreadsheet: Field 22', 'Spreadsheet: Field 23',\
#                        'Spreadsheet: Field 24', 'Spreadsheet: Field 25', 'Spreadsheet: Field 26', 'Spreadsheet: Field 27', 'Spreadsheet: Field 28',\
#                        'Spreadsheet: Field 29', 'Spreadsheet: Field 30', 'Spreadsheet: Field 31', 'Spreadsheet: Field 32', 'DM',\
#                        'DM Anti', 'EF', 'EF Anti', 'RP', 'RP Anti', 'RP Ctx1', 'RP Ctx2', 'WM', 'WM Anti', 'WM Ctx1', 'WM Ctx1', 'Spreadsheet: display', 'UTC Date and Time']]
#
# # Define batch size and create batches
# incrementList = [i + 1 for i, name in enumerate(opened_xlsxFile_selection['Component Name']) if name == 'Fixation Timing']
# numberBatches = len(incrementList) // batchLength
#
# # Split the data into batches based on the fixation timing component
# for batchNumber in range(numberBatches):
#     batchOn = batchNumber * batchLength
#     batchOff = batchNumber * batchLength + batchLength
#     numFixStepsTotal = 0
#     numRespStepsTotal = 0
#
#     # Calculate average fix, resp and total steps for this batch
#     for j in incrementList[batchOn:batchOff]:
#         # Accumulate step numbers
#         currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
#         numFixSteps = round(currentTrial['Onset Time'][0] / 20)
#         numRespSteps = round(currentTrial['Onset Time'][1] / 20)
#         numFixStepsTotal += numFixSteps
#         numRespStepsTotal += numRespSteps
#         # Calculate average after cumulative addition of whole batch
#         if j == incrementList[batchOff - 1]:
#             numFixStepsAverage = int(numFixStepsTotal / batchLength)
#             numRespStepsAverage = int(numRespStepsTotal / batchLength)
#             totalStepsAverage = numFixStepsAverage + numRespStepsAverage
#             # print(numFixStepsAverage,numRespStepsAverage,totalStepsAverage)
#
#     finalSequenceList = []
#     # Create sequences of every trial in the current batch with the previous calculated time steps
#     for j in incrementList[batchOn:batchOff]:
#         currentTrial = opened_xlsxFile_selection[j:j + 2].reset_index().drop(columns=['index'])
#         currentSequenceList = []
#         for k in range(totalStepsAverage):
#             sequence = [currentTrial.iloc[0]]
#             currentSequenceList.append(sequence)
#         finalSequenceList.append(currentSequenceList)
#
#     # Create final df for INPUT and OUPUT
#     newOrderSequenceList = []
#     # Append all the time steps accordingly to a list
#     for j in range(0, totalStepsAverage):
#         for i in range(0, len(finalSequenceList)):
#             newOrderSequenceList.append(finalSequenceList[i][j])
#
#     # Create Yang form
#     finalTrialsList_array = np.array(newOrderSequenceList).reshape((len(finalSequenceList[0]), len(finalSequenceList), 87))
#     # Create meta dict before deleting necessary information from current trials List
#     date = str(finalTrialsList_array[0,0,86]).split(' ')[0]
#     sleepingQualityQuestion, sleepingQualityValue, drugVectorQuestion, drugVectorValue = find_sleepingQuality_drugVector(opened_questionare, date)
#     print('# DEBUGGING ###########################################################################################')
#     print('>>>>>>>>>>>>>>>>>>>>>>>>     ', taskString, final_questionaire, date)
#     print('>>>>>>>>>>>>>>>>>>>>>>>>     ', 'sleepingQualityValue:', sleepingQualityValue, 'drugVector:',drugVectorValue)
#     print('# DEBUGGING ###########################################################################################')
#     # Catch meta data now that you have all the necessary
#     meta_dict = {'date_time': date, 'difficultyLevel': finalTrialsList_array[0,0,0],
#                  'timeLimit': finalTrialsList_array[0,0,1], 'sleepingQualityQuestion': sleepingQualityQuestion,
#                  'sleepingQualityValue': sleepingQualityValue, 'drugVectorQuestion': drugVectorQuestion,
#                  'drugVectorValue': drugVectorValue}
#     # Create one input file and one output file
#     Input, Output = finalTrialsList_array, finalTrialsList_array
#     Input = np.delete(Input, [0, 1, 2, 3, 4, 5, 6, 8, 85,86], axis=2) # todo: 77 statt 85 ????
#     Output = np.delete(Output, np.s_[0, 1, 2, 4, 5, 6, 7,86], axis=2)
#     Output = np.delete(Output, np.s_[34:78], axis=2)
#     Input, Output = Input[:, sequence_on:sequence_off, :], Output[:, sequence_on:sequence_off, :]
#
#     # INPUT ############################################################################################################
#     # float all fixation input values to 1
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             Input[i][j][0] = float(1)
#     # float all task values to 0
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             for k in range(65, Input.shape[2]):
#                 Input[i][j][k] = float(0)
#
#     # Define a task dictionary for specific task-related columns
#     taskDict = {'DM': 65, 'DM Anti': 66, 'EF': 67, 'EF Anti': 68, 'RP': 69, 'RP Anti': 70, 'RP Ctx1': 71,
#                 'RP Ctx2': 72, 'WM': 73, 'WM Anti': 74, 'WM Ctx1': 75, 'WM Ctx2': 76}
#
#     # float all task values to 1
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             if len(opened_xlsxFile_selection['Spreadsheet'][0].split('_')) == 4:
#                 Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]]] = float(1)
#                 taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0]
#             else:
#                 Input[i][j][taskDict[opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + ' ' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]]] = float(1)
#                 taskShorts = opened_xlsxFile_selection['Spreadsheet'][0].split('_')[0] + '_' + opened_xlsxFile_selection['Spreadsheet'][0].split('_')[1]
#
#     # float all 000_000's on field units to 0
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             for k in range(1, 65):
#                 if Input[i][j][k] == '000_000.png':
#                     Input[i][j][k] = float(0)
#
#     # Define modulation dictionaries for specific columns
#     mod1Dict = {'red': 0.08, 'rust': 0.17, 'orange': 0.25, 'amber': 0.33, 'yellow': 0.42, 'lime': 0.5, 'green': 0.58, \
#                 'moss': 0.66, 'blue': 0.75, 'violet': 0.83, 'magenta': 0.92, 'purple': 1.0}
#
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             for k in range(1, 33):
#                 if Input[i][j][k] != 0:
#                     Input[i][j][k] = mod1Dict[Input[i][j][k].split('_')[0]]
#
#     # Define modulation dictionaries for specific columns
#     mod2Dict = {'triangle.png': float(0.2), 'pentagon.png': float(0.4), 'heptagon.png': float(0.6), 'nonagon.png': float(0.8), 'circle.png': float(1.0)}
#
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             for k in range(33, 65):
#                 if Input[i][j][k] != 0:
#                     Input[i][j][k] = mod2Dict[Input[i][j][k].split('_')[1]]
#
#     # float all field values of fixation period to 0
#     for i in range(0, numFixStepsAverage):
#         for j in range(0, Input.shape[1]):
#             for k in range(1, 65):
#                 Input[i][j][k] = float(0)
#
#     # Add input gradient activation
#     # Create default hyperparameters for network
#     num_ring, n_eachring, n_rule = 2, 32, 12
#     n_input, n_output = 1 + num_ring * n_eachring + n_rule, n_eachring + 1
#     pref = np.arange(0, 2 * np.pi, 2 * np.pi / 32)
#
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             currentTimeStepModOne = Input[i][j][1:33]
#             currentTimeStepModTwo = Input[i][j][33:65]
#             # Allocate first unit ring
#             unitRingMod1 = np.zeros(32, dtype='float32')
#             unitRingMod2 = np.zeros(32, dtype='float32')
#
#             # Get non-zero values of time steps on both modalities and also directly fix distance between units issue
#             NonZero_Mod1 = np.nonzero(currentTimeStepModOne)[0]
#             NonZero_Mod2 = np.nonzero(currentTimeStepModTwo)[0]
#             if len(NonZero_Mod1) != 0:
#                 # Accumulating all activities for both unit rings together
#                 for k in range(0, len(NonZero_Mod1)):
#                     currentStimLoc_Mod1 = pref[NonZero_Mod1[k]]
#                     currentStimStrength_Mod1 = currentTimeStepModOne[NonZero_Mod1[k]]
#                     currentStimLoc_Mod2 = pref[NonZero_Mod2[k]]
#                     currentStimStrength_Mod2 = currentTimeStepModTwo[NonZero_Mod2[k]]
#                     # add one gradual activated stim to final form
#                     currentActivation_Mod1 = add_x_loc(currentStimLoc_Mod1, pref) * currentStimStrength_Mod1
#                     currentActivation_Mod2 = add_x_loc(currentStimLoc_Mod2, pref) * currentStimStrength_Mod2
#                     # Add all activations for one trial together
#                     unitRingMod1 = np.around(unitRingMod1 + currentActivation_Mod1, decimals=2)
#                     unitRingMod2 = np.around(unitRingMod2 + currentActivation_Mod2, decimals=2)
#                 # Store
#                 currentFinalRow = np.concatenate((Input[i][j][0:1], unitRingMod1, unitRingMod2, Input[i][j][65:78]))
#                 Input[i][j][0:78] = currentFinalRow
#
#     # Change dtype of every element in matrix to float32 for later validation functions
#     for i in range(0, Input.shape[0]):
#         for j in range(0, Input.shape[1]):
#             for k in range(0, Input.shape[2]):
#                 Input[i][j][k] = np.float32(Input[i][j][k])
#     # Also change dtype for entire array
#     Input = Input.astype('float32')
#
#     # Save input data
#     os.chdir(os.path.join(os.getcwd(), main_path, taskShorts))
#     input_filename = (participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
#                       xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Input')
#     np.save(input_filename, Input)
#
#     # Sanity check
#     print('Input solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])
#
#     # OUTPUT ###########################################################################################################
#     # float all field units during fixation epoch on 0.05
#     for i in range(0, numFixStepsAverage):
#         for j in range(0, Output.shape[1]):
#             for k in range(2, 34):
#                 Output[i][j][k] = float(0.05)
#     # float all field units of response epoch to 0
#     for i in range(numFixStepsAverage, totalStepsAverage):
#         for j in range(0, Output.shape[1]):
#             for k in range(2, 34):
#                 Output[i][j][k] = float(0)
#     # float all fixation outputs during response period to 0.05
#     for i in range(numFixStepsAverage, totalStepsAverage):
#         for j in range(0, Output.shape[1]):
#             Output[i][j][1] = float(0.05)
#
#     # Assign field units to their according participant response value after fixation period
#     outputDict_RP_1 = {'Image 2': 2, 'Image 4': 4, 'Image 6': 6, 'Image 8': 8, 'Image 10': 10, 'Image 12': 12, 'Image 14': 14,\
#         'Image 16': 16, 'Image 18': 18, 'Image 20': 20, 'Image 22': 22, 'Image 24': 24, 'Image 26': 26, 'Image 28': 28, 'Image 30': 30, 'Image 32': 32}
#
#     outputDict_RP_2 = {'Image 1': 1, 'Image 3': 3, 'Image 5': 5, 'Image 7': 7, 'Image 9': 9, 'Image 11': 11, 'Image 13': 13, 'Image 15': 15,\
#         'Image 17': 17, 'Image 19': 19, 'Image 21': 21, 'Image 23': 23, 'Image 25': 25, 'Image 27': 27, 'Image 29': 29, 'Image 31': 31}
#
#     for i in range(numFixStepsAverage, totalStepsAverage):
#         for j in range(0, Output.shape[1]):
#             # Get the right dictionary
#             if Output[i][j][34].split(' RP')[0] == 'Display 1':
#                 outputDict = outputDict_RP_1
#             elif Output[i][j][34].split(' RP')[0] == 'Display 2':
#                 outputDict = outputDict_RP_2
#
#             if Output[i][j][0] != 'screen' and Output[i][j][0] != 'noResponse' and Output[i][j][0] != 'NoResponse':
#                 Output[i][j][outputDict[Output[i][j][0]]] = np.float32(0.85)
#             else:
#                 for k in range(2, 34):
#                     Output[i][j][k] = np.float32(0.05)
#
#     # Drop unnecessary columns
#     Output = np.delete(Output, [0,34], axis=2)
#     # Pre-allocate y-loc matrix; needed for later validation
#     y_loc = np.zeros((Output.shape[0], Output.shape[1]))
#
#     # Add output gradient activation
#     for i in range(0, Output.shape[0]):
#         for j in range(0, Output.shape[1]):
#             currentTimeStepOutput = Output[i][j][1:33]
#             # Allocate first unit ring
#             unitRingOutput = np.zeros(32, dtype='float32')
#             # Get non-zero values of time steps
#             nonZerosOutput = np.nonzero(currentTimeStepOutput)[0]
#             # Float first fixations rows with -1
#             for k in range(0, numFixStepsAverage):
#                 y_loc[k][j] = np.float(-1)
#
#             if len(nonZerosOutput) == 1:
#                 # Get activity and model gradient activation around it
#                 currentOutputLoc = pref[nonZerosOutput[0]]
#                 currentActivation_Output = add_x_loc(currentOutputLoc, pref) + 0.05  # adding noise
#                 unitRingOutput = np.around(unitRingOutput + currentActivation_Output, decimals=2)
#                 # Store
#                 currentFinalRow = np.concatenate((Output[i][j][0:1], unitRingOutput))
#                 Output[i][j][0:33] = currentFinalRow
#                 # Complete y_loc matrix
#                 for k in range(numFixStepsAverage, totalStepsAverage):
#                     y_loc[k][j] = pref[nonZerosOutput[0]]
#
#     # Change dtype of every element in matrix to float32 for later validation functions
#     for i in range(0, Output.shape[0]):
#         for j in range(0, Output.shape[1]):
#             for k in range(0, Output.shape[2]):
#                 Output[i][j][k] = np.float32(Output[i][j][k])
#     # Also change dtype for entire array
#     Output = Output.astype('float32')
#
#     # Save output data
#     output_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
#                       xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Output'
#     np.save(output_filename, Output)
#     # Save y_loc data
#     yLoc_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
#                     xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'yLoc'
#     np.save(yLoc_filename, y_loc)
#     # Save meta data
#     meta_filename = participant + '-' + 'month_' + str(month) + '-' + 'batch_' + str(batchNumber) + '-' + taskShorts + '-' + \
#                     xlsxFile.split('_')[3].split('-')[0] + '_' + xlsxFile.split('_')[3].split('-')[1] + '-' + 'Meta'
#     with open('{}.json'.format(meta_filename), 'w') as json_file:
#         json.dump(meta_dict, json_file)
#
#     # Sanity check
#     print('Meta solved:', 'sleepingQualityValue: ', meta_dict['sleepingQualityValue'])
#     print('Output solved: ', opened_xlsxFile_selection['Spreadsheet'][0], ' ', opened_xlsxFile_selection['TimeLimit'][0])
