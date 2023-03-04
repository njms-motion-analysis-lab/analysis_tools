# -*- coding: utf-8 -*-

import csv
import os
from typing import Any
import numpy as np
import sys
import constants
from patient import Patient
import re
import pandas as pd
from csv_generator import CSVGenerator
from exp_motion import ExpMotion
from exp_motion_sample_trial import ExpMotionSampleTrial
from exp_motion_sample import ExpMotionSample


# path = "/Users/stephenmacneille/Documents/labs/controls_filteredandtrimmed/block/filteredAndTrimmed_S001_Block_dominant.npy"

part = constants.PARTS[sys.argv[1]]
side = constants.SIDES[sys.argv[2]]
sensor = constants.SENSORS[sys.argv[3]]
dimension = sys.argv[4]

obj = eval(f'constants.{part}.Position.{side}.{sensor}')
motions = obj.DIMENSIONS[dimension.capitalize()]

root_dir = "controls_alignedCoordinateSystem"
exp_motion_sample_sets = {}
for subdir, dirs, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith('.npy'):
            v = ExpMotionSample(file_path, motions)

            exp_motion_sample_sets[v.name] = v
            name = os.path.splitext(file_path)[0]


patients = {}
new_exp_motion_sample_sets = {}
exp_motions = {}

for key, value in exp_motion_sample_sets.items():
    sub_dir = key.split('_')
    root = sub_dir[0]
    patient = sub_dir[1]
    exp_motion = sub_dir[2]
    varient = sub_dir[-1]

    if varient != exp_motion:
        exp_motion = exp_motion + '_' + varient

    if exp_motion.endswith('_'):
        exp_motion = exp_motion[:-1]

    if exp_motion in exp_motions.keys():
        exp_motions[exp_motion].add_sample(value)
    else:
        m = ExpMotion(exp_motion)
        m.add_sample(value)
        exp_motions[exp_motion] = m

    if patient in patients.keys():
        patients[patient].add_exp_motions(value)
    else:
        p = Patient(patient)
        p.add_exp_motions(value)
        patients[patient] = p

exp_stats = {}
for k, v in exp_motions.items():
    v.set_stats()
    exp_stats[v.name] = [v.mean, v.stdev]

p_stats = {}
for k, pt in patients.items():
    pt.set_stats()
    p_stats[pt.name] = [pt.mean, pt.stdev]


csv_name = CSVGenerator.camel_to_snake(part + side + sensor + '_'+ motions[-1] + ".csv")
CSVGenerator.generate_csv(exp_motions, patients, csv_name)
