from multiprocessing import freeze_support
import os
import re
import sqlite3
from creation_services.old_generator import OldGenerator
from models.legacy_sensor import Sensor
from models.legacy_cohort import Cohort
from migrations.legacy_table import Table
import pdb

import pandas as pd
import numpy as np
import os

RAW_DATA_FOLDER = "raw_data/analysis_me"

if __name__ == '__main__':
    freeze_support()
    exp = {}

    def generate_sensors():
        Sensor.find_or_create(
            name = "TimeStamp",
            kind = "(epoch)",
            side = 'na',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mAccelerometerMagnitude",
            kind = "(m/s^2)",
            side = 'na',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mAccelerometerX",
            kind = "(m/s^2)",
            side = 'x',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mAccelerometerY",
            kind = "(m/s^2)",
            side = 'y',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mAccelerometerZ",
            kind = "(m/s^2)",
            side = 'z',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mGyroscopeMagnitude",
            kind = "(rad/s)",
            side = 'na',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mGyroscopeX",
            kind = "(rad/s)",
            side = 'x',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mGyroscopeY",
            kind = "(rad/s)",
            side = 'y',
            part = 'na',
            placement = 'a',
        )
    
        Sensor.find_or_create(
            name = "mGyroscopeZ",
            kind = "(rad/s)",
            side = 'z',
            part = 'na',
            placement = 'a',
        )
        print("done!")



    
    files = []

    def convert_csv_and_preserve_headers(csv_file):
        try:
            # Attempt to read the CSV file with the default UTF-8 encoding
            df = pd.read_csv(csv_file)
        except UnicodeDecodeError:
            # If UTF-8 decoding fails, try reading the file with 'ISO-8859-1' encoding
            print(f"UTF-8 decoding failed for {csv_file}. Trying with 'ISO-8859-1' encoding.")
            df = pd.read_csv(csv_file, encoding='ISO-8859-1')

        # Generate the .npy file path by replacing the .csv extension with .npy
        npy_file = csv_file.rsplit('.', 1)[0] + '.npy'
        
        # Check and delete the existing .npy file if it exists
        if os.path.exists(npy_file):
            os.remove(npy_file)
            print(f'Deleted old {npy_file}')
        
        # Convert the DataFrame into a structured array to preserve headers
        structured_arr = df.to_records(index=False)
        
        # Save the structured array to a .npy file
        np.save(npy_file, structured_arr)
        
        print(f'Saved {csv_file} as {npy_file}')
        return npy_file

    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)

            if file_path.lower().endswith('.npy'):
                files.append(file_path)
                print(file_path)
            elif file_path.lower().endswith('.csv'):
                file_path = convert_csv_and_preserve_headers(file_path)
                files.append(file_path)
                

    n = 0

    # generate_sensors()




    for file in files:
        # Modify this `if` statement to select the files of choice.
        if "analysis_me" in file.lower():
            print("hey")
            if "group" in file.lower():
                print(file.lower())
                parts = file.split('/')
                cohort = f"{parts[2]}_{parts[1].replace(' ', '_')}"
                cohort = re.sub(r'(?<!^)(?=[A-Z])', '_', cohort).lower().replace(' ', '_')
                cohort = Cohort.find_or_create(name=cohort, is_control=False, is_treated=False)
                OldGenerator(file, cohort, skip_list=True)
        else:
            print("nope", file.lower())
        

print("Done!")



# path = raw_data/analysis_me/group 1/s019/balance/task 1_trial 2.npy

# cohort = "analysis_me_group_1", 
# patient_name = "amg1_S019"
# task = "Balance01"
# trial = 2

# path = raw_data/analysis_me/group 1/s019/gait/task 2_trial 1.npy

# cohort = "analysis_me_group_1", 
# patient_name = "amg1_S019"
# task = "Gait02"
# trial = 1

# path = raw_data/analysis_me/group 2/s017/gait/task 1_trial 1.npy

# cohort = "analysis_me_group_2", 
# patient_name = "amg2_S017"
# task = "Gait01"
# trial = 1


# raw_data/analysis_me/group 2/s017/gait/task 1_trial 1.npy
# raw_data/analysis_me/group 2/s021/balance/task 2_trial 1.npy
# raw_data/analysis_me/group 2/s021/balance/task 2_trial 1.npy
# raw_data/analysis_me/group 2/s021/balance/task 2_trial 3.npy

