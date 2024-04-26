from multiprocessing import freeze_support
import os
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

    def convert_csv(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Convert the DataFrame into a NumPy array
        data = df.values
        
        # Generate the .npy file path by replacing the .csv extension with .npy
        npy_file = csv_file.rsplit('.', 1)[0] + '.npy'
        
        # Save the NumPy array to a .npy file
        np.save(npy_file, data)
        
        print(f'Saved {csv_file} as {npy_file}')
    # Dictionary to track filenames and their paths
    files_dict = {}

    # Walk through the directory structure
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            
            # Add to dictionary or append if duplicate
            if filename in files_dict:
                files_dict[filename].append(file_path)
            else:
                files_dict[filename] = [file_path]

    # Iterate over the dictionary to find duplicates and decide which to delete
    for filename, paths in files_dict.items():
        if len(paths) > 1:
            # Here you can implement your logic to decide which file to keep.
            # The simplest approach is to keep the first one and delete the rest.
            # Modify this logic as per your requirement.
            for file_path in paths[1:]:  # Skip the first one, delete the rest
                os.remove(file_path)
                print(f"Deleted duplicate file: {file_path}")

    # Now process files that are not duplicates or the ones chosen to be kept
    for _, paths in files_dict.items():
        for file_path in paths[:1]:  # Process only the first (or remaining) file
            if file_path.lower().endswith('.csv'):
                convert_csv(file_path)
            else:
                # Here you can handle non-CSV files if needed
                pass

    generate_sensors()




    for file in files:
        

        print(file.lower())
        # Modify this `if` statement to select the files of choice.
        if "analysis_me" in file.lower():
            if "group" in file.lower():
                print(file.lower())
                # cohort = Cohort.find_or_create(name="analysis_me", is_control=False, is_treated=False)
        else:
            print("nope", file.lower())
        

print("Done!")

