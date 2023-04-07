import sqlite3

from table import Table
import pdb

Table.create_tables()

from models.base_model import BaseModel
from models.sensor import Sensor
from models.patient import Patient


pdb.set_trace()

print("Bye!")