from multiprocessing import freeze_support
import os
import sqlite3
from creation_services.old_generator import OldGenerator
from models.legacy_sensor import Sensor
from models.legacy_cohort import Cohort
from migrations.legacy_table import Table
import pdb

RAW_DATA_FOLDER = "raw_data/analysis_me"

if __name__ == '__main__':
    freeze_support()
    exp = {}
    root_dir = "controls_alignedCoordinateSystem"

    def create_sensor_from_string(sensor_string):
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        part = sensor_string[1:3]
        
        placement = sensor_string[3]
        if len(sensor_string) > 6:
            placement = sensor_string[3:5]
        axis = sensor_string[-1]
        Sensor.find_or_create(name=sensor_string, side=side, axis=axis, placement=placement, part=part)

    def generate_sensors():
        # sensors = []
        sensors = [
    
    TimeStamp
    kind = "(epoch)"
    
    mAccelerometerMagnitude
    kind = "(m/s^2)"
    
    mAccelerometerX
    kind = "(m/s^2)"
    
    mAccelerometerY
    kind = "(m/s^2)"
    
    mAccelerometerZ
    kind = "(m/s^2)"
    
    mGyroscopeMagnitude
    kind = "(rad/s)"
    
    mGyroscopeX
    kind = "(rad/s)"
    
    mGyroscopeY
    kind = "(rad/s)"
    
    mGyroscopeZ
    kind = "(rad/s)"
            ""

        ]
        for s_string in sensors:
            create_sensor_from_string(s_string)

    # generate_sensors()
    files = []
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith('.npy'):
                files.append(file_path)

    n = 0
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

