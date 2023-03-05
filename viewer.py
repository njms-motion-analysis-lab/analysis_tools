import os
import sys
from typing import Any

import numpy as np
import pandas as pd

import constants
from csv_generator import CSVGenerator
from exp_motion import ExpMotion
from exp_motion_sample import ExpMotionSample
from patient import Patient


part = constants.PARTS[sys.argv[1]]
side = constants.SIDES[sys.argv[2]]
sensor = constants.SENSORS[sys.argv[3]]
dimension = sys.argv[4]

obj = eval(f'constants.{part}.Position.{side}.{sensor}')
motions = obj.DIMENSIONS[dimension.capitalize()]

root_dir = "controls_alignedCoordinateSystem"
exp_motion_sample_sets = {}

for subdir, _, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith('.npy'):
            v = ExpMotionSample(file_path, motions)
            exp_motion_sample_sets[v.name] = v

exp_motions = {}
patients = {}

for key, value in exp_motion_sample_sets.items():
    sub_dir = key.split('_')
    root, patient, exp_motion, varient = sub_dir[0], sub_dir[1], sub_dir[2], sub_dir[-1]

    if varient != exp_motion:
        exp_motion = exp_motion + '_' + varient

    if exp_motion.endswith('_'):
        exp_motion = exp_motion[:-1]

    if exp_motion in exp_motions:
        exp_motions[exp_motion].add_sample(value)
    else:
        m = ExpMotion(exp_motion)
        m.add_sample(value)
        exp_motions[exp_motion] = m

    if patient in patients:
        patients[patient].add_exp_motions(value)
    else:
        p = Patient(patient)
        p.add_exp_motions(value)
        patients[patient] = p

exp_stats = {k: [v.mean, v.stdev] for k, v in exp_motions.items()}
p_stats = {k: [v.mean, v.stdev] for k, v in patients.items()}

csv_name = CSVGenerator.camel_to_snake(f"{part}{side}{sensor}_{motions[-1]}.csv")
CSVGenerator.generate_csv(exp_motions, patients, csv_name)