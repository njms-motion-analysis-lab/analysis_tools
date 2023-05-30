import sqlite3
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel

from migrations.legacy_table import LegacyTable
import pdb
import re

from models.task import Task

from models.gradient_set import GradientSet
from models.sensor import Sensor
import os

def replace_axis_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            
            # Read the current title
            with open(filepath, "rb") as file:
                content = file.read()
                title_start = content.find(b"tEXtTitle")
                title_end = content.find(b"\x00", title_start)
                title = content[title_start+9:title_end].decode("latin-1")
            
            # Replace the axis portion of the title
            new_title = title.replace("x", "x (anterior-posterior)").replace("y", "y (medial-lateral)").replace("z", "z (superior-inferior)")
            
            # Modify the title in the PNG file
            modified_content = content[:title_start+9] + new_title.encode("latin-1") + content[title_end:]
            
            # Save the modified PNG file
            with open(filepath, "wb") as file:
                file.write(modified_content)



# dir = 'parallel_plots/grad_data__kurtosis'
# replace_axis_labels(dir)
# for t in tasks:
#     t.gen_all_stats_csv(abs_val=False, non_normed=True)


from prediction_tools.predictor import Predictor
from models.task import Task
from models.sensor import Sensor

LegacyTable.update_tables()
from models.task import Task
t = Task.all()[2]


# Retrieve all instances of the Trial class
all_trials = Task.all()

# Iterate through each trial instance
# for task in all_trials:
#     if "Dominant" in task.description and "Nondominant" not in task.description:
#         task.update(is_dominant=True)
#     if "Nondominant" in task.description or "nondominant" in task.description:
#         task.update(is_dominant=False)
#     if "dominant" in task.description and "nondominant" not in task.description:
#         task.update(is_dominant=True)

#     for trial in task.trials():
#         trial.update(is_dominant=task.is_dominant)

print("done")

import pdb;pdb.set_trace()

# dt = Task.dominant()

# dtt = dt[0]
# print("task", dtt.description)
# snr = Sensor.all()[7]
# print("sensor:", snr.name)
# pr = Predictor(dtt,snr)

# print()
# # non normalized
# print("Normalized...")
# acc = pr.train_it(non_norm=False, abs_val=False)
# print("-------------------")
# print("Non normalized...")
# acc = pr.train_it(non_norm=True, abs_val=False)
# print("-------------------")
# print("Non normalized and absolute values...")
# acc = pr.train_it(non_norm=True, abs_val=True)
# print("-------------------")


# #
# #  neural network
# acc = pr.train_it_nn(non_norm=False, abs_val=False)
# acc = pr.train_it_nn(non_norm=True, abs_val=False)

# import pdb;
# pdb.set_trace()


# print("Done!") 