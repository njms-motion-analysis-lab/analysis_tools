import sqlite3

from table import Table
import pdb

Table.create_tables()

from models.base_model import BaseModel
from models.sensor import Sensor
from models.patient import Patient
from models.patient_task import PatientTask
from models.gradient_set import GradientSet
from models.task import Task
sensor = Sensor.where(id=37)[0]



items = GradientSet.all()[3].get_aggregate_stats().index
for it in items:
    print(it)

pdb.set_trace()


print("Bye!")