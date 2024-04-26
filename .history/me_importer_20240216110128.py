import asyncio
from datetime import datetime
import os
import concurrent.futures
from models.sensor import Sensor
from creation_services.legacy_generator import LegacyGenerator
import os
import numpy as np
from migrations.table import Table
from tsfresh import extract_features  # or your own custom feature extraction function

RAW_DATA_FOLDER = "raw_data/analysis_me"


SENSORS = [
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


# placeholder for feature extraction function
def extract_features_from_model(model):
    # replace this with actual feature extraction code
    return extract_features(model)

async def main():
    # Table.remove_deadlock()
    # Table.clear_tables()
    # Table.drop_all_tables()
    # Table.create_tables()
    # from models.patient_task import PatientTask
    # from models.patient import Patient
    # from models.gradient_set import GradientSet
    # from models.sub_gradient import SubGradient
    # Table.update_tables()

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
        for s_string in SENSORS:
            create_sensor_from_string(s_string)

    generate_sensors()
    
    
    

    # get a list of npy file paths
    files = []
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith('.npy'):
                files.append(file_path)
    
    # Create LegacyGenerator objects concurrently
    
    n = 0
    for file in files:
        if "alignedCoordsByStart" not in file.lower():
            if "ring" in file.lower() or "block" in file.lower():
                print(file)
                LegacyGenerator(file)
                n += 1
                print(file,"finished", n, "features complete")
            else:
                print("skipping", file)
        
        
    # Create an instance of LegacyGenerator
    # legacy_generator = LegacyGenerator()

        
        # futures_io = [loop.run_in_executor(pool, LegacyGenerator, file) for file in files]  # use process pool here
        
        # wait for all IO-bound tasks to complete and get the results (the models)
        # models = await asyncio.gather(*futures_io)

        # perform feature extraction concurrently
        # futures_cpu = [loop.run_in_executor(pool, extract_features_from_model, model) for model in models]

        # wait for all CPU-bound tasks to complete and get the results (the extracted features)
        # features = await asyncio.gather(*futures_cpu)

        # print(features)  # or do something else with the features

if __name__ == "__main__":
    asyncio.run(main())