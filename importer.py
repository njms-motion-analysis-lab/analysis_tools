import os
import sqlite3
from generator import Generator
from models.sensor import Sensor
from table import Table
import pdb

exp = {}
root_dir = "controls_alignedCoordinateSystem"

# Call the function to create the tables before you start using the Generator class
Table.drop_all_tables()
Table.create_tables()

def create_sensor_from_string(sensor_string):
    side_map = {"l": "left", "r": "right"}
    side = side_map.get(sensor_string[0], None)
    
    part = sensor_string[1:3]
    placement = sensor_string[3]
    axis = sensor_string[-1]
    sensor = Sensor.find_or_create(name=sensor_string, side=side, axis=axis, placement=placement, part=part)



def generate_sensors():
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
        "lfrm_x",
        "lfrm_y",
        "lfrm_z",
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
    ]
    for s_string in sensors:
        create_sensor_from_string(s_string)


from models.task import Task
from models.sensor import Sensor
from models.patient import Patient
from models.position_set import PositionSet

from models.patient_task import PatientTask

from models.gradient_set import GradientSet
from models.trial import Trial

generate_sensors()
print("done 1")
for subdir, _, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith('.npy'):
            v = Generator(file_path)
            exp[v.name] = v



        # ls = Sensor.all()[0]
        # ps = PositionSet.all()[0]
        # gs = GradientSet.all()[0]
        # # mat = gs.get_matrix('matrix')
        # gs.get_sub_tasks()

        
import pdb
pdb.set_trace()

Table.drop_all_tables()
print("Done!")
