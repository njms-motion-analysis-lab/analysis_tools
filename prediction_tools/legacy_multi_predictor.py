# import sqlite3
import json
import pickle
import pandas as pd
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from models.legacy_sensor import Sensor
from prediction_tools.legacy_predictor import Predictor


SENSOR_CODES = [
    'rbhd_x',
    'relb_x',
    'relbm_x',
    'rfhd_x',
    'rfin_x',
    'rfrm_x',
    'rsho_x',
    'rupa_x',
    'rwra_x',
    'rwrb_x',
]


class MultiPredictor(LegacyBaseModel):
    table_name = "multi_predictor"

    def __init__(self, id=None, task_id=None, sensors=Sensor.where(name=SENSOR_CODES), model=RandomForestClassifier(), items = None, created_at=None, updated_at=None, cohort_id=None):
        super().__init__()
        self.id = id
        self.task_id = task_id
        self.sensors = sensors
        self.model = model
        self.items = {}
        self.cohort_id = cohort_id

    def gen_items_for_sensor(self, snr, ntaf=True):
        if not ntaf:
            print(f"Generating normed, abs score, for task: {self.task.description}")
            nfat = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=True, cohort_id=self.cohort_id)
            nfat.train_it()
            print(f"Generating normed, non abs score, for task: {self.task.description}")
            nfaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=False, cohort_id=self.cohort_id)
            nfaf.train_it()
            print(f"Generating non normed, abs score, for task: {self.task.description}")
            ntat = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=True, abs_val=True, cohort_id=self.cohort_id)
            # ntat.train_it()
            print(f"Generating non normed, non abs score, for task: {self.task.description}", cohort_id=self.cohort_id)
        
        ntaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=True, abs_val=True, cohort_id=self.cohort_id)
        ntaf.train_it()
        return ntaf
    
    def get_sensors(self):
        return Sensor.where(name=SENSOR_CODES)
    
    def get_predictors(self):
        return Predictor.where(multi_predictor_id=self.id)
    
    def load_items(self):
        with open('items.pickle', 'rb') as handle:
            self.items = pickle.load(handle)
    
    def gen_scores_for_sensor(self):
        for sensor in self.get_sensors():
            predictor = Predictor.find_or_create(task_id=self.task_id, sensor_id=sensor.id, non_norm=True, abs_val=False, multi_predictor_id=self.id, cohort_id=self.cohort_id)
            # skip this sensor as its data isnt great atm.
            if predictor.sensor().name != 'relbm_x':
                predictor = predictor.retrain_from(force_load=True)
                import pdb;pdb.set_trace()
            # Here we assume that the accuracy is stored in predictor.accuracies after training
            # print(f"accuracy: {predictor.accuracies}")

            # self.items[sensor.name] = predictor.accuracies
        # Save items to disk
        with open('items.pickle', 'wb') as handle:
            pickle.dump(self.items, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("done!")
   
    