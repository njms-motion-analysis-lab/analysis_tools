import os
import sqlite3
from generator import Generator
from models.base_model import BaseModel
from models.sensor import Sensor
from models.patient import Patient
from table import Table
import pdb

exp = {}
root_dir = "controls_alignedCoordinateSystem"

# Call the function to create the tables before you start using the Generator class
Table.create_tables()

import pdb

pdb.set_trace()

for subdir, _, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith('.npy'):
            v = Generator(file_path)
            exp[v.name] = v

Table.clear_tables()
print("Done!")
