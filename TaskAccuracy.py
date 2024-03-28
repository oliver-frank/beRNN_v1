import os
import pandas as pd
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)

########################################################################################################################
# TaskAccuracy
########################################################################################################################
# Participant list
participant_dir = 'Z:\Desktop\ZI\PycharmProjects\BeRNN\Data'
participantList = os.listdir(participant_dir)

participant = participantList[4] # choose which particpant to analyze
month = '4' # choose which month to analyze

percentCorrect_DM, count_DM = 0, 0
percentCorrect_DM_Anti, count_DM_Anti = 0, 0
percentCorrect_EF, count_EF = 0, 0
percentCorrect_EF_Anti, count_EF_Anti = 0, 0
percentCorrect_RP, count_RP = 0, 0
percentCorrect_RP_Anti, count_RP_Anti = 0, 0
percentCorrect_RP_Ctx1, count_RP_Ctx1 = 0, 0
percentCorrect_RP_Ctx2, count_RP_Ctx2 = 0, 0
percentCorrect_WM, count_WM = 0, 0
percentCorrect_WM_Anti, count_WM_Anti = 0, 0
percentCorrect_WM_Ctx1, count_WM_Ctx1 = 0, 0
percentCorrect_WM_Ctx2, count_WM_Ctx2 = 0, 0

# co: Download data as .xlsx long format
list_testParticipant_month = os.listdir(os.path.join(participant_dir,participant,month))
for i in list_testParticipant_month:
    if i.split('.')[1] != 'png':
        currentFile = pd.read_excel(os.path.join(participant_dir,participant,month,i), engine='openpyxl')
        # print(currentFile['UTC Date and Time'])
        if isinstance(currentFile.iloc[0,28],float) == False: # avoid first rows with state questions .xlsx files
            # print(currentFile.iloc[0,28].split('_trials_')[0])
            # print('W:/AG_CSP/Projekte/BeRNN/02_Daten/BeRNN_main/' + participant + month + i)
            if currentFile.iloc[0,28].split('_trials_')[0] == 'DM':
                # percentCorrect_DM += currentFile['Store: PercentCorrectDM'][len(currentFile['Store: PercentCorrectDM'])-3]
                # count_DM += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_DM += sum(filtered_rows['Store: PercentCorrectDM'])
                count_DM += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'DM_Anti':
                # percentCorrect_DM_Anti += currentFile['Store: PercentCorrectDMAnti'][len(currentFile['Store: PercentCorrectDMAnti'])-3]
                # count_DM_Anti += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_DM_Anti += sum(filtered_rows['Store: PercentCorrectDMAnti'])
                count_DM_Anti += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'EF':
                # percentCorrect_EF += currentFile['Store: PercentCorrectEF'][len(currentFile['Store: PercentCorrectEF'])-3]
                # count_EF += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_EF += sum(filtered_rows['Store: PercentCorrectEF'])
                count_EF += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'EF_Anti':
                # percentCorrect_EF_Anti += currentFile['Store: PercentCorrectEFAnti'][len(currentFile['Store: PercentCorrectEFAnti'])-3] # no extra displays for Anti were made
                # count_EF_Anti += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_EF_Anti += sum(filtered_rows['Store: PercentCorrectEFAnti'])
                count_EF_Anti += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'RP':
                # percentCorrect_RP += currentFile['Store: PercentCorrectRP'][len(currentFile['Store: PercentCorrectRP'])-3]
                # count_RP += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 124].copy() # todo: every now and then the 125th event is missing
                percentCorrect_RP += sum(filtered_rows['Store: PercentCorrectRP'])
                count_RP += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Anti':
                # percentCorrect_RP_Anti += currentFile['Store: PercentCorrectRPAnti'][len(currentFile['Store: PercentCorrectRPAnti'])-3]
                # count_RP_Anti += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_RP_Anti += sum(filtered_rows['Store: PercentCorrectRPAnti'])
                count_RP_Anti += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Ctx1':
                # percentCorrect_RP_Ctx1 += currentFile['Store: PercentCorrectRPCtx1'][len(currentFile['Store: PercentCorrectRPCtx1'])-3]
                # count_RP_Ctx1 += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 124].copy() # todo: every now and then the 125th event is missing
                percentCorrect_RP_Ctx1 += sum(filtered_rows['Store: PercentCorrectRPCtx1'])
                count_RP_Ctx1 += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'RP_Ctx2':
                # percentCorrect_RP_Ctx2 += currentFile['Store: PercentCorrectRPCtx2'][len(currentFile['Store: PercentCorrectRPCtx2'])-3]
                # count_RP_Ctx2 += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_RP_Ctx2 += sum(filtered_rows['Store: PercentCorrectRPCtx2'])
                count_RP_Ctx2 += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'WM':
                # percentCorrect_WM += currentFile['Store: PercentCorrectWM'][len(currentFile['Store: PercentCorrectWM'])-3]
                # count_WM += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_WM += sum(filtered_rows['Store: PercentCorrectWM'])
                count_WM += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Anti':
                # percentCorrect_WM_Anti += currentFile['Store: PercentCorrectWMAnti'][len(currentFile['Store: PercentCorrectWMAnti'])-3]
                # count_WM_Anti += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_WM_Anti += sum(filtered_rows['Store: PercentCorrectWMAnti'])
                count_WM_Anti += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Ctx1':
                # percentCorrect_WM_Ctx1 += currentFile['Store: PercentCorrectWMCtx1'][len(currentFile['Store: PercentCorrectWMCtx1'])-3]
                # count_WM_Ctx1 += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_WM_Ctx1 += sum(filtered_rows['Store: PercentCorrectWMCtx1'])
                count_WM_Ctx1 += len(filtered_rows)
                print('currentFile processed')

            if currentFile.iloc[0,28].split('_trials_')[0] == 'WM_Ctx2':
                # percentCorrect_WM_Ctx2 += currentFile['Store: PercentCorrectWMCtx2'][len(currentFile['Store: PercentCorrectWMCtx2'])-3]
                # count_WM_Ctx2 += 1

                filtered_rows = currentFile[currentFile['Event Index'] == 125].copy()
                percentCorrect_WM_Ctx2 += sum(filtered_rows['Store: PercentCorrectWMCtx2'])
                count_WM_Ctx2 += len(filtered_rows)
                print('currentFile processed')

acc_DM = percentCorrect_DM/count_DM
acc_DM_Anti = percentCorrect_DM_Anti/count_DM_Anti
acc_EF = percentCorrect_EF/count_EF
acc_EF_Anti = percentCorrect_EF_Anti/count_EF_Anti
acc_WM = percentCorrect_WM/count_WM
acc_WM_Anti = percentCorrect_WM_Anti/count_WM_Anti
acc_WM_Ctx1 = percentCorrect_WM_Ctx1/count_WM_Ctx1
acc_WM_Ctx2 = percentCorrect_WM_Ctx2/count_WM_Ctx2
acc_RP = percentCorrect_RP/count_RP
acc_RP_Anti = percentCorrect_RP_Anti/count_RP_Anti
acc_RP_Ctx1 = percentCorrect_RP_Ctx1/count_RP_Ctx1
acc_RP_Ctx2 = percentCorrect_RP_Ctx2/count_RP_Ctx2

# pd.DataFrame(data={'acc_WM_Ctx2':[acc_WM_Ctx2]})

########################################################################################################################
# Plot training effect
########################################################################################################################
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import matplotlib.dates as mdates

# Participant list
participant_dir = 'Z:\Desktop\ZI\PycharmProjects\BeRNN\Data'
participantList = os.listdir(participant_dir)
participant = participantList[0] # choose which particpant to analyze
month = '4' # choose which month to analyze

# Specify the folder containing the .xlsx files
folder_path = os.path.join(participant_dir,participant,month)

# Define filenames and corresponding colors
filename_color_dict = {
    'DM': 'red', 'DM_Anti': 'orange',
    'EF': 'blue', 'EF_Anti': 'darkblue',
    'RP': 'green', 'RP_Anti': 'darkgreen',
    'RP_Ctx1': 'limegreen', 'RP_Ctx2': 'forestgreen',
    'WM': 'yellow', 'WM_Anti': 'gold',
    'WM_Ctx1': 'lemonchiffon', 'WM_Ctx2': 'darkkhaki'
}

# Initialize empty lists to store combined x and y values
all_x_values = []
all_y_values = []

for task in filename_color_dict:
    print(task, filename_color_dict[task])

    # Create right name for ycolumn
    ycolumn = 'Store: PercentCorrect' + ''.join(task.split('_'))

    # Initialize empty lists to store combined x and y values
    all_x_values = []
    all_y_values = []

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".xlsx"):
            file_path = os.path.join(folder_path, filename)

            # Load the Excel file into a DataFrame
            df = pd.read_excel(file_path, engine='openpyxl')
            if isinstance(df.iloc[0, 28], float) == False and df.iloc[0, 28].split('_trials_')[0] == task:
                try:
                    # Filter rows where "Event Index" is 125
                    filtered_rows = df[df['Event Index'] == 125].copy()
                    print(filename)

                    # Convert "Date and Time" to datetime format where possible
                    filtered_rows['Local Date and Time'] = pd.to_datetime(filtered_rows['Local Date and Time'], errors='coerce')
                    # Extract values from "Date and Time" and "Accuracy" columns
                    x_values = pd.to_datetime(filtered_rows['Local Date and Time'].dt.strftime('%d-%m-%Y'))
                    y_values = filtered_rows[ycolumn]

                    print('x_values: ', x_values)
                    print('y_values: ', y_values)

                    # Append values to the combined lists
                    all_x_values.extend(x_values)
                    all_y_values.extend(y_values)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    # Sort the all_x_values
    all_x_values.sort()

    # Plot the task-related data
    plt.scatter(all_x_values, all_y_values, color=filename_color_dict[task])

    # Calculate linear regression
    all_x_values = mdates.date2num(all_x_values)
    slope, intercept, r_value, p_value, std_err = linregress(all_x_values, all_y_values)

    # Plot the regression line
    regression_line = slope * np.array(all_x_values) + intercept
    plt.plot(all_x_values, regression_line, color=filename_color_dict[task],
             label=task + ' ' + f'Regression (R^2={r_value ** 2:.2f})')

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.xlabel('Date and Time')
plt.ylabel('Accuracy')
plt.title('Training effect: ' + participant + '_' + month)
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility if needed

# Let matplotlib autoscale the x-axis
plt.autoscale(enable=True, axis='x')

# Save the figure to the folder where the data is from
figure_path = os.path.join(folder_path, participant + '_' + month + '_' + 'Training_Effect.png')
plt.savefig(figure_path, bbox_inches='tight')

plt.tight_layout()
plt.show()
