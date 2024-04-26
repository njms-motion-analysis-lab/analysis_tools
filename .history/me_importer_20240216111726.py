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
            "lwra_x",
            "lwrb_x",
            "lwra_y",
            "lwrb_y",
            "lwra_z",
            "lwrb_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "rfrm_x",
            "rfrm_y",
            "rfrm_z",
            "lelb_x",
            "lelbm_x",
            "lelb_y",
            "lelbm_y",
            "lelb_z",
            "lelbm_z",
            "relb_x",
            "relbm_x",
            "relb_y",
            "relbm_y",
            "relb_z",
            "relbm_z",
            "lupa_x",
            "lupa_y",
            "lupa_z",
            "rupa_x",
            "rupa_y",
            "rupa_z",
            "lsho_x",
            "lsho_y",
            "lsho_z",
            "rsho_x",
            "rsho_y",
            "rsho_z",
            "lwra_x",
            "lwrb_x",
            "lwra_y",
            "lwrb_y",
            "lwra_z",
            "lwrb_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "lfrm_x",
            "lfrm_y",
            "lfrm_z",
            'lfhd_x',
            'lfhd_y',
            'lfhd_z',
            'rfhd_x', 
            'rfhd_y', 
            'rfhd_z', 
            'lbhd_x',
            'lbhd_y', 
            'lbhd_z', 
            'rbhd_x', 
            'rbhd_y', 
            'rbhd_z',
            'rfin_x', 
            'rfin_y', 
            'rfin_z',
            'lfin_x', 
            'lfin_y', 
            'lfin_z',
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
        if "alignedcoordsbystart" in file.lower():
            if "cp" in file.lower():
                cohort = Cohort.find_or_create(name="cp_before", is_control=False, is_treated=False)

            
            if "block" in file.lower() and "dynamic" not in file.lower():
                print(file)
                OldGenerator(file, cohort)
                n += 1
                print(file,"finished", n, "features complete")
            else:
                print("skipping", file)
        else:
            print("nope", file.lower())
        

print("Done!")

