# Function for sorting out already augmented files #####################################################################
import os
import datetime

def get_creation_time(file_path):
    # Get the creation time of the file
    if os.name == 'nt':  # For Windows
        creation_time = os.path.getctime(file_path)
    else:  # For Unix-like systems
        stat = os.stat(file_path)
        try:
            creation_time = stat.st_birthtime
        except AttributeError:
            # For Linux, use last metadata change time as creation time is not available
            creation_time = stat.st_mtime
    return datetime.datetime.fromtimestamp(creation_time)

def is_file_newer_than(file_path, date_time):
    creation_time = get_creation_time(file_path)
    return creation_time > date_time

# Example usage
file_path = 'path/to/your/file.txt'
date_time = datetime.datetime(2023, 5, 1, 12, 0, 0)  # Replace with your specific date and time

if is_file_newer_than(file_path, date_time):
    print(f"The file {file_path} is newer than {date_time}.")
    # Your commands here
else:
    print(f"The file {file_path} is older than {date_time}.")
    # Your other commands here





# Error_Comparison #####################################################################################################
# Define necessary variables at

