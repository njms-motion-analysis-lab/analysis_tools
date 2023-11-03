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

    @classmethod
    def combo_train(self, pred_1, pred_2, classifier_name):
        print('umm,')
        df1 , y1 = pred_1.get_final_bdf()
        df2 , y2 = pred_1.get_final_bdf()
        combo_df= MultiPredictor().create_composite_dataframe(df1, df2)
        print('start')
        classifier_accuracies, classifier_params, classifier_metrics = {}, {}, {}
        print("hey")
        classifiers = Predictor.define_classifiers_cls(False, False, classifier_name=classifier_name)
        groups = combo_df['patient']
        scores = {}
        params = {}
        acc_metrics = {}
        # Check to make sure classifiers is a dictionary
        if classifiers is not None:
            if 'classifier' in classifiers and 'param_grid' in classifiers:
                # We have a single classifier's configuration, not a dictionary of classifiers
                classifier_data = classifiers  # The current dictionary is actually the data
                classifier_name = classifier_name
                scores = {}
                params = {}
                acc_metrics = {}
                import pdb; pdb.set_trace()
                classifier_scores, best_params, extra_metrics, scores, params, acc_metrics = Predictor._train_from_mp(combo_df, y1, groups, classifier_name, classifier_data, scores, params, acc_metrics, use_shap=True, mps=self)
            # Existing code where you use the loop
        else:
            
            print('yo')


        



    @classmethod
    def create_composite_dataframe(cls, df1, df2):
        # First, ensure that the 'patient' and 'is_dominant' columns are of the same data type in both DataFrames
        df1['patient'] = df1['patient'].astype(int)
        df1['is_dominant'] = df1['is_dominant'].astype(int)
        df2['patient'] = df2['patient'].astype(int)
        df2['is_dominant'] = df2['is_dominant'].astype(int)

        # Perform an inner join on 'patient' and 'is_dominant' columns
        return pd.merge(df1, df2, on=['patient', 'is_dominant'], suffixes=('_task1', '_task2'))