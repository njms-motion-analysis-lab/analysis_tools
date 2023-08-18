from multiprocessing import freeze_support
import os
import sqlite3
from creation_services.old_generator import OldGenerator
from models.legacy_sensor import Sensor
from migrations.legacy_table import Table
import pdb
RAW_DATA_FOLDER = "raw_data/controls_alignedCoordinateSystem"
if __name__ == '__main__':
    freeze_support()
    exp = {}
    root_dir = "controls_alignedCoordinateSystem"

    # Call the function to create the tables before you start using the LegacyGenerator class
    # Table.drop_all_tables()
    # Table.update_tables()
    def create_sensor_from_string(sensor_string):
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        # if sensor_string is 'lelbm_x':
        #     import pdb
        #     pdb.set_trace()
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

    generate_sensors()
    print("done 1")
    # from models.dynamic_sub_gradient import DynamicSubGradient
    # from models.dynamic_position_set import DynamicPositionSet
    # from models.dynamic_gradient_set import DynamicGradientSet
    # DynamicSubGradient.delete_all()
    # DynamicGradientSet.delete_all()
    # DynamicPositionSet.delete_all()
    # Table.drop_all_tables()
    # Table.clear_tables()
    # Table.create_tables()
    # Table.update_tables()
    files = []
    
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith('.npy'):
                files.append(file_path)

    n = 0
    for file in files:
        print(file.lower())
        if "alignedcoordsbystart" in file.lower():
            if "block" in file.lower():
                print(file)
                OldGenerator(file)
                n += 1
                print(file,"finished", n, "features complete")
            else:
                print("skipping", file)
        else:
            print("nope", file.lower())
        

print("Done!")

