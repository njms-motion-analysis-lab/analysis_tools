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
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith('.csv'):
                files.append(file_path)

    n = 0

    generate_sensors()

    def convert(csv_file):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)
        
        # Convert the DataFrame into a NumPy array
        data = df.values
        
        # Generate the .npy file path by replacing the .csv extension with .npy
        npy_file = csv_file.rsplit('.', 1)[0] + '.npy'
        
        # Save the NumPy array to a .npy file
        np.save(npy_file, data)
        
        print(f'Saved {csv_file} as {npy_file}')


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

