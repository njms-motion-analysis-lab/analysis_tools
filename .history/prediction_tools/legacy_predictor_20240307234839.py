import json
import pickle
from collections import Counter
import string

import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_cohort import Cohort
from models.legacy_patient import Patient
from models.legacy_patient_task import PatientTask
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from boruta import BorutaPy
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from prediction_tools.predictor_score import PredictorScore

print(tf.__version__)
from catboost import CatBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, LeaveOneOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from skrebate import ReliefF
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from ts_fresh_params import PARAMS, get_params_for_column
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SHAP_CLASSIFIERS = [
    'XGBoost', 'RandomForest', 'DecisionTree', 'ExtraTrees', 'CatBoost', 
    'GradientBoosting'
]

NUMBER_OF_FEATURES = 10
DEFAULT_K_FOLD_SPLITS = 5

COMPATIBLE_MODELS = [
    "DecisionTreeClassifier", 
    "RandomForestClassifier", 
    "ExtraTreesClassifier", 
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "CatBoostClassifier"
]

DEFAULT_FEATURES = ['grad_data__sum_values','grad_data__abs_energy','grad_data__mean_abs_change', 'grad_data__mean_change', 'grad_data__mean_second_derivative_central', 'grad_data__variation_coefficient','grad_data__standard_deviation','grad_data__skewness','grad_data__kurtosis','grad_data__variance','grad_data__root_mean_square','grad_data__mean', 'grad_data__length']
MINIMUM_SAMPLE_SIZE = 15
MINI_PARAMS = {
    'RandomForest': {
        'classifier': RandomForestClassifier(),
        'param_grid': {
            'classifier__n_estimators': [1, 2, 3, 4, 5],
            'classifier__max_depth': [1, 2, 3],
            'classifier__min_samples_split': [2, 3],
            'classifier__min_samples_leaf': [1, 2, 3,],
            'classifier__max_features': ['sqrt', 'log2']
        }
    },
    'KNN': {
        'classifier': KNeighborsClassifier(),
        'param_grid': {
            'classifier__n_neighbors': [1, 2, 3, 4],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
        }
    },
    'LogisticRegression': {
        'classifier': LogisticRegression(max_iter=10000, solver='liblinear'),
        'param_grid': {
            'classifier__C': [0.01, 0.1, 1],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear', 'saga']
        }
    },
    'XGBoost': {
        'classifier': XGBClassifier(),
        'param_grid': {
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [1, 2, 3],
            'classifier__max_depth': [1, 2]
        }
    },
    'CatBoost': {
        'classifier': CatBoostClassifier(verbose=0),
        'param_grid': {
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__iterations': [1, 2, 3],
            'classifier__depth': [1, 2]
        }
    },
    'SVM': {
        'classifier': SVC(probability=True),
        'param_grid': {
            'classifier__C': [0.1, 1],
            'classifier__gamma': ['scale', 0.1],
            'classifier__kernel': ['linear', 'poly']
        }
    },
    'AdaBoost': {
        'classifier': AdaBoostClassifier(),
        'param_grid': {
            'classifier__n_estimators': [1, 2, 3],
            'classifier__learning_rate': [0.01, 0.1]
        }
    },
    'ExtraTrees': {
        'classifier': ExtraTreesClassifier(),
        'param_grid': {
            'classifier__n_estimators': [1, 2, 3],
            'classifier__max_depth': [1, 2],
            'classifier__min_samples_split': [2],
            'classifier__min_samples_leaf': [1, 2]
        }
    },
    'GradientBoosting': {
        'classifier': GradientBoostingClassifier(),
        'param_grid': {
            'classifier__learning_rate': [0.01, 0.1],
            'classifier__n_estimators': [1, 2, 3],
            'classifier__max_depth': [1, 2],
            'classifier__subsample': [0.8, 1.0],
            'classifier__max_features': ['sqrt', 'log2']
        }
    },
    'DecisionTree': {
        'classifier': DecisionTreeClassifier(),
        'param_grid': {
            'classifier__max_depth': [1, 2],
            'classifier__min_samples_split': [2],
            'classifier__min_samples_leaf': [1, 2],
            'classifier__max_features': ['sqrt', 'log2', None]
        }
    },
}

class CustomKerasClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, batch_size=32, epochs=50, dropout_rate=0.2):
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate

    def fit(self, X, y):
        self.model = self._build_model(X.shape[1])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=0)
        return self

    def _build_model(self, n_features):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(n_features,)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        return model

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype("int32")

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))
    
    def predict_proba(self, X):
        # If your model outputs a single probability,
        # you might need to shape the output like this:
        probabilities = self.predict(X)
        return np.hstack([1 - probabilities, probabilities])

class Predictor(LegacyBaseModel):
    table_name = "predictor"

    def __init__(self, id=None, task_id=None, sensor_id=None, non_norm=False, abs_val=False, accuracies={}, matrix=None, created_at=None, updated_at=None, multi_predictor_id=None, aggregated_stats=None, aggregated_stats_non_normed=None, cohort_id=None):
        self.id = id
        self.task_id = task_id
        self.sensor_id = sensor_id
        self.non_norm = non_norm
        self.abs_val = abs_val
        self.accuracies = accuracies
        self.matrix = matrix
        self.created_at = created_at
        self.updated_at = updated_at
        self.aggregated_stats_non_normed = aggregated_stats_non_normed
        self.aggregated_stats = aggregated_stats
        self.multi_predictor_id = multi_predictor_id
        self.cohort_id = cohort_id
        self.skip_boruta = False
    
    def sensor(self):
        return Sensor.get(self.sensor_id)
    
    def task(self):
        return Task.get(self.task_id)
    
    def cohort(self):
        return Cohort.get(self.cohort_id)

    def alt_sensor(self, sensor=None, cohort=None):
        # quick way to check cohort
        

        if sensor is not None:
            sensor = Sensor.get(sensor.id)
        else:
            sensor = Sensor.get(self.sensor_id)
        if self.cohort_id == 4 or self.cohort_id == 5 or self.cohort_id == 3:
            # keeping the same name
            return sensor
        task = Task.get(self.task_id)
        nondom_sensor_name = task.get_counterpart_sensor(sensor.name)
        nondom = Sensor.find_by("name", nondom_sensor_name)
        return nondom

    def select_features(self):
        sensor = Sensor.get(self.sensor_id)
        nondom_sensor = self.alt_sensor()
        if (self.non_norm is True) or (self.abs_val is True) or (self.non_norm == 1) or (self.abs_val == 1):
            features_dom = []
            features_non_dom = []
            for loc in DEFAULT_FEATURES:
                features_dom.append(loc.replace('grad_data', sensor.name))
                features_non_dom.append(loc.replace('grad_data', nondom_sensor.name))
        else:
            features_dom = []
            features_non_dom = []
            for loc in DEFAULT_FEATURES:
                features_dom.append(loc)
                features_non_dom.append(loc)

        return [features_dom, features_non_dom]

    def get_aggregated_stats_non_normed(self):
        return pickle.loads(self.aggregated_stats_non_normed)
    
    def get_aggregated_stats(self):
        return pickle.loads(self.aggregated_stats)

    def correct_attributes(self):
        created_at = self.aggregated_stats_non_normed
        updated_at = self.aggregated_stats
        aggregated_stats_non_normed = self.updated_at
        aggregated_stats = None
        
        print(f"self", self.multi_predictor_id)
        print(
            created_at,
            updated_at,
            str(aggregated_stats_non_normed)[:200],
            aggregated_stats
        )


        self.update(
            aggregated_stats_non_normed = aggregated_stats_non_normed,
            aggregated_stats = aggregated_stats,
            created_at = created_at,
            updated_at=updated_at
        )

        # self.save
        print("done", self.multi_predictor_id, self.id)
        
    def get_top_n_correlated_columns(self, column_name, n):
        # First, fetch the DataFrame
        df = self.get_df()

        # Check if the column exists in the DataFrame
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")

        # Calculate the correlation matrix
        correlation_matrix = df.corr()

        # Get the absolute values of correlations for the specified column
        correlations = correlation_matrix[column_name].abs()

        # Sort the correlations and get the top n columns with their correlation coefficients
        top_n = correlations.sort_values(ascending=False)[1:n+1]

        # Return a list of tuples with column names and their correlation coefficients
        return top_n

    def is_alt_compare(self):
        return Cohort.get(self.cohort_id).is_alt_compare()

    def generate_sub_gradient_count(self, sensor=None):
        dom_task = Task.get(self.task_id)
        
        if sensor is None:
            sensor = Sensor.get(self.sensor_id)
    
        nondom_sensor = self.alt_sensor(sensor=sensor)
        self_pts = PatientTask.where(task_id=dom_task.id, cohort_id=self.cohort_id)

        dom_dataframes = []
        nondom_dataframes = []
        features_dom = self.select_features()[0]
        features_non_dom = self.select_features()[1]

        other_pts = []
        if self.cohort().is_alt_compare():
            counterpart_sensor = sensor
            compare_cohort = self.cohort().get_alt_cohort()
            other_pts = PatientTask.where(task_id=dom_task.id, cohort_id=compare_cohort.id)
        else:
            counterpart_task = dom_task.get_counterpart_task()[0]
            compare_cohort = self.cohort()
            counterpart_sensor = nondom_sensor

        for self_pt in self_pts:
            if len(other_pts) != 0:
                counterpart_pts = other_pts.pop()
            else:
                if self.cohort().is_alt_compare():
                    counterpart_pts = []
                else:
                    counterpart_pts = PatientTask.where(task_id=counterpart_task.id, patient_id=self_pt.patient_id, cohort_id=compare_cohort.id)
            counterpart_sensor = nondom_sensor
            
            if not counterpart_pts:
                    continue
            counterpart_pt = counterpart_pts[0]
            curr_patient = Patient.where(id=self_pt.patient_id)[0].name
            self_temp, counter_temp = self.get_temp_sub_gradients_count(self_pt, counterpart_pt, sensor, counterpart_sensor, curr_patient, features_dom, features_non_dom)
            if self_pt.patient_id == 21 or self_pt.patient_id == 26 or self_pt.patient_id == 27 or self_pt.patient_id == 28:
                print("switching places for lefty pt....")
                dom_dataframes.append(counter_temp)
                nondom_dataframes.append(self_temp)

            else:
                dom_dataframes.append(self_temp)
                nondom_dataframes.append(counter_temp)
        
        dom_df = pd.concat(dom_dataframes)
        nondom_df = pd.concat(nondom_dataframes)

        df = pd.concat([dom_df, nondom_df])
        df_reset = df.reset_index()
        name = 'grad_data__mean_sub_gradient_count' + '_' + sensor.axis
        df_renamed = df_reset.rename(columns={'index': 'name', 0: name})
        renamed = df_renamed.drop(columns=['name'])

        
        return renamed
    
    def get_temp_sub_gradients_count(self, self_pt, counterpart_pt, sensor, counterpart_sensor, curr_patient, features_dom=False, features_non_dom=False):
        def get_temp_sub_gradient_count(patient_pt, patient_sensor, features_loc):
            stats_method = abs if self.abs_val == 1 else lambda x: x
            
            temp_df = pd.DataFrame(
                stats_method(
                    patient_pt.combined_gradient_set_count(
                        patient_sensor, abs_val=self.abs_val, non_normed=self.non_norm, loc=features_loc
                    )
                )
            ).T
            return temp_df
        self_temp = get_temp_sub_gradient_count(self_pt, sensor, False)
        try:
            counter_temp = get_temp_sub_gradient_count(counterpart_pt, counterpart_sensor, False)
        except TypeError:
            print("error for patient task pt:", counterpart_pt, "sensor:", counterpart_sensor.id)
            print("using other hand stat")
            counter_temp = self_temp

        is_dominant = curr_patient != 'S017'
        self_temp['is_dominant'] = is_dominant
        counter_temp['is_dominant'] = not is_dominant

        for temp in [self_temp, counter_temp]:
            temp['patient'] = curr_patient
        return self_temp, counter_temp

    def get_default_sensor(self):
        return Sensor.get(self.sensor_id)

    def get_tasks(self):
        dom_task = Task.get(self.task_id)
        if self.cohort().is_alt_compare():
            counterpart_task = dom_task
        else:
            counterpart_task = dom_task.get_counterpart_task()[0]
        return dom_task, counterpart_task

    def get_cohorts_and_sensors(self, sensor):
        if self.cohort().is_alt_compare():
            compare_cohort = self.cohort().get_alt_cohort()
            counterpart_sensor = sensor
        else:
            compare_cohort = self.cohort()
            counterpart_sensor = self.alt_sensor(sensor=sensor)
        return compare_cohort, counterpart_sensor

    def is_lefty_patient(self, patient_id):
        return patient_id in [21, 26, 27, 28]
    
    # an alternative method of processing PT where we are not pairing individual PTs on the basis of a patient

    def process_alt_tasks(self, self_pts, other_pts, sensor, nondom_sensor):
        dom_dataframes = []
        nondom_dataframes = []

        num = 0
        def get_single_patient_dataframe(self_pt, sensor, curr_patient, cohort, is_dominant=True):
            self_temp = self.get_temp_dataframe(self_pt, sensor, False)
            if self_temp is None:
                return None
            self_temp['is_dominant'] = is_dominant
            self_temp['cohort'] = cohort.name
            self_temp['patient'] = curr_patient

        
        for self_pt in self_pts:
            curr_patient = Patient.where(id=self_pt.patient_id)[0].name
            cohort = Cohort.get(self.cohort_id)
            self_temp = get_single_patient_dataframe(self_pt, sensor, curr_patient, cohort, is_dominant=True)
            print(self_temp)
            if self_temp is not None:
                dom_dataframes.append(self_temp)
                num+=1
        num = 0
        for other_pt in other_pts:
            other_patient = Patient.get(other_pt.patient_id).name
            other_cohort = self.cohort().get_alt_cohort()
            other_temp = get_single_patient_dataframe(other_pt, sensor, other_patient, other_cohort, is_dominant=True)
            print(other_temp)
            if other_temp is not None:
                nondom_dataframes.append(other_temp)
                num+=1

        return dom_dataframes, nondom_dataframes


    def process_patient_tasks(self, self_pts, other_pts, dom_task, counterpart_task, compare_cohort, sensor, nondom_sensor, force_old=False):
        if self.cohort().is_alt_compare() is True and force_old is False:
            return self.process_alt_tasks(self_pts, other_pts, sensor, nondom_sensor)
    
        dom_dataframes = []
        nondom_dataframes = []
        features_dom, features_non_dom = self.select_features()

        for self_pt in self_pts:
            counterpart_pts = self.get_counterpart_pts(other_pts, counterpart_task, self_pt, compare_cohort)
            if not counterpart_pts:
                continue

            curr_patient = Patient.where(id=self_pt.patient_id)[0].name
            self_temp, counter_temp = self.get_temp_dataframes(self_pt, counterpart_pts[0], sensor, nondom_sensor, curr_patient, features_dom, features_non_dom)

            if self.is_lefty_patient(self_pt.patient_id):
                print("Switching places for lefty patient...")
                dom_dataframes.append(counter_temp)
                nondom_dataframes.append(self_temp)
            else:
                if self_temp is not None:
                    dom_dataframes.append(self_temp)
                if counter_temp is not None:
                    nondom_dataframes.append(counter_temp)

        return dom_dataframes, nondom_dataframes


    def alt_compare(self, self_pts, other_pts, dom_task, counterpart_task, compare_cohort, sensor, nondom_sensor):
        dom_dataframes = []
        nondom_dataframes = []
        features_dom, features_non_dom = self.select_features()

        # Convert other_pts to a dictionary for easier access
        other_pts_dict = {pt.patient_id: pt for pt in other_pts}

        for self_pt in self_pts:
            # Attempt to find a counterpart in other_pts_dict using the patient_id
            counterpart_pt = other_pts_dict.get(self_pt.patient_id)

            if not counterpart_pt:
                # If no counterpart is found, you may decide to skip this self_pt
                # or handle it differently depending on your requirements
                continue

            curr_patient = Patient.where(id=self_pt.patient_id)[0].name
            self_temp, counter_temp = self.get_temp_dataframes(self_pt, counterpart_pt, sensor, nondom_sensor, curr_patient, features_dom, features_non_dom)

            if self.is_lefty_patient(self_pt.patient_id):
                print("Switching places for lefty patient...")
                dom_dataframes.append(counter_temp)
                nondom_dataframes.append(self_temp)
            else:
                dom_dataframes.appencd(self_temp)
                nondom_dataframes.append(counter_temp)

        return dom_dataframes, nondom_dataframes

    def get_counterpart_pts(self, other_pts, counterpart_task, self_pt, compare_cohort):
        if other_pts:
            return [other_pts.pop()]
        elif not self.cohort().is_alt_compare():
            return PatientTask.where(task_id=counterpart_task.id, patient_id=self_pt.patient_id, cohort_id=compare_cohort.id)
        return []

    def generate_dataframes(self, sensor=None, add_other=False):
        sensor = sensor or self.get_default_sensor()
        
        dom_task, counterpart_task = self.get_tasks()
        self_pts = PatientTask.where(task_id=dom_task.id, cohort_id=self.cohort_id)
        compare_cohort, nondom_sensor = self.get_cohorts_and_sensors(sensor)
        other_pts = PatientTask.where(task_id=dom_task.id, cohort_id=compare_cohort.id) if self.cohort().is_alt_compare() else []
        dom_dataframes, nondom_dataframes = self.process_patient_tasks(self_pts, other_pts, dom_task, counterpart_task, compare_cohort, sensor, nondom_sensor, force_old=True)


        print(dom_dataframes)
        print(nondom_dataframes)

        dom_dataframes = [df.reset_index(drop=True) for df in dom_dataframes if not df.empty]
        nondom_dataframes = [df.reset_index(drop=True) for df in nondom_dataframes if not df.empty]

        # Check if there are any DataFrames to concatenate, to avoid errors when lists are empty
        if dom_dataframes:
            concatenated_dom = pd.concat(dom_dataframes, ignore_index=True)
        else:
            concatenated_dom = pd.DataFrame()  # Create an empty DataFrame if no dom_dataframes

        if nondom_dataframes:
            concatenated_nondom = pd.concat(nondom_dataframes, ignore_index=True)
        else:
            concatenated_nondom = pd.DataFrame()  # Create an empty DataFrame if no nondom_dataframes
        



        # Concatenate the dom and nondom dataframes, ensuring ignore_index=True is applied
        
        final_df = pd.concat([concatenated_dom, concatenated_nondom], ignore_index=True)

        def with_extra_cohort():
            compare_cohort_b = compare_cohort.get_alt_cohort()
            other_pts = PatientTask.where(task_id=dom_task.id, cohort_id=compare_cohort_b.id)
            dom_dataframes_b, nondom_dataframes_b = self.process_patient_tasks(self_pts, other_pts, dom_task, counterpart_task, compare_cohort, sensor, nondom_sensor, force_old=True)
            if nondom_dataframes_b:
                concatenated_nondom_b = pd.concat(nondom_dataframes_b, ignore_index=True)
            else:
                concatenated_nondom_b = pd.DataFrame()  # Create an empty DataFrame if no nondom_dataframes
            return pd.concat([final_df, concatenated_nondom_b], ignore_index=True)

        if add_other is True:
            return with_extra_cohort()
        
        return final_df


    def get_temp_dataframe(self, patient_pt, patient_sensor, features_loc):
        print(patient_sensor.attrs())
        stats_method = abs if self.abs_val == 1 else lambda x: x
        gradient_stats = patient_pt.combined_gradient_set_stats_list(patient_sensor,abs_val=self.abs_val,non_normed=self.non_norm,loc=features_loc)
        # Check if 'mean' is in the gradient stats and process accordingly
        if 'mean' in gradient_stats:
            temp_df = pd.DataFrame(
                stats_method(gradient_stats['mean'])
            ).T
        else:
            # Initialize an empty DataFrame with an expected structure if 'mean' is not present
            sen = self.sensor()
            gs = patient_pt.get_gradient_sets_for_sensor(sen)
            temp_df = pd.DataFrame(columns=['grad_data__placeholder'])
            print("No 'mean' data available for the given gradient set.")
            return None

            
        
        temp_df.columns = ['grad_data__' + col.split('__')[1] for col in temp_df.columns]
        return temp_df
    

    def get_temp_dataframes(self, self_pt, counterpart_pt, sensor, counterpart_sensor, curr_patient, features_dom=False, features_non_dom=False):
        print("getting temp df abs" if self.abs_val == 1 else "getting temp df norm")

        print("self...")
        self_temp = self.get_temp_dataframe(self_pt, sensor, False)
        print("counter...")
        counter_temp = self.get_temp_dataframe(counterpart_pt, counterpart_sensor, False)

        if not self.is_alt_compare():
            is_dominant = curr_patient != 'S017'
        else:
            is_dominant = True
        cohort = Cohort.get(self_pt.cohort_id)
        counter_cohort = Cohort.get(counterpart_pt.cohort_id)
        
        if counter_temp is not None:
            counter_temp['cohort'] = counter_cohort.name
            counter_temp['is_dominant'] = not is_dominant
            counter_temp['patient'] = Patient.get(counterpart_pt.patient_id)
    
        if self_temp is not None:
            self_temp['cohort'] = cohort.name
            self_temp['is_dominant'] = is_dominant    
            self_temp['patient'] = curr_patient
            

        return self_temp, counter_temp

    def rename_duplicate_columns(self, df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df

    def match_holdout_columns_to_bdf(self, bdf):
        """
        Ensure that the holdout dataset has the exact same columns as the provided bdf.
        If there are extra columns in the holdout dataset, they are dropped.
        If there are missing columns in the holdout dataset, they are added with default NaN values.
        """
        # Drop columns that are in the holdout_data but not in bdf
        columns_to_drop = [column for column in self.holdout_data.columns 
                        if column not in bdf.columns and column != 'is_dominant']
        if columns_to_drop:
            self.holdout_data = self.holdout_data.drop(columns_to_drop, axis=1)
        
        # Add columns that are in bdf but missing in holdout_data
        for column in bdf.columns:
            if column not in self.holdout_data.columns:
                self.holdout_data[column] = np.nan  # Using NaN as default value, you can replace this if you have a better default value

        # To ensure the order of columns in holdout_data matches bdf
        if 'is_dominant' in self.holdout_data:
            y_holdout = self.holdout_data['is_dominant']
            self.holdout_data = self.holdout_data[bdf.columns]
            self.holdout_data['is_dominant'] = y_holdout
        else:
            self.holdout_data = self.holdout_data[bdf.columns]
    
    def train_from(self, obj=None, use_shap=False, force_load=False, get_sg_count=False):
        if obj is None:
            obj = self

        bdf, y = self.get_final_bdf(untrimmed=True, force_load=force_load, get_sg_count=get_sg_count)

        bdf = Predictor.trim_bdf_with_boruta(bdf, y)
        if len(bdf.columns) <= 2:
            print("Skipping boruta!")
            self.skip_boruta == len(bdf.columns)

            bdf, y = self.get_final_bdf(force_load=False, get_sg_count=get_sg_count)
        self.fit_multi_models(bdf, y, use_shap)
        print("Done training!")
    
    def retrain_from(self, use_shap=False, force_load=False, get_sg_count=False):
        # only use if best dataframe saved on predictor
        bdf = self.get_df(force_load=force_load)
        
        y = self.get_y(bdf)

        self.fit_multi_models(bdf, y, use_shap)

    def get_final_bdf(self, force_load=False, force_abs_x=False, untrimmed=False, get_sg_count=False):
        bdf = self.get_df(force_load=force_load)
        if get_sg_count is True:
            other = self.generate_sub_gradient_count()
            bdf = pd.merge(other, bdf, on=['is_dominant', 'patient'], how='inner')
        y = self.get_y(bdf)
        

        if force_abs_x == True:
            columns_to_drop = [col for col in bdf.columns if col.endswith('_x')]
            bdf.drop(columns=columns_to_drop, inplace=True)

        bdf['is_dominant'] = y
        is_dominant = bdf['is_dominant'].copy()

        bdf = self.rename_columns_with_orientation(bdf)
        bdf = self.clean_bdf(bdf) 

        print("curr", bdf)


        # return early untrimmed bdf 
        if untrimmed:
            print(bdf)
            return bdf, y
        
        bdf_t = self.trim_bdf(bdf, custom_limit=50)

        
        
        bdf_t['is_dominant'] = is_dominant

        return [bdf_t, y]

    @classmethod
    def trim_bdf_with_boruta(cls, bdf, y, n_estimators=500, max_depth=5, random_state=42):
        # Assuming 'bdf' is your dataframe without the target variable 'is_dominant'
        # Columns to retain regardless of feature selection
        columns_to_retain = ['is_dominant', 'patient']
        columns_to_retain = [col for col in columns_to_retain if col in bdf.columns]


        # Separate features and target
        X = bdf.drop(columns=columns_to_retain)

        # Initialize Random Forest
        rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

        # Initialize Boruta
        boruta_selector = BorutaPy(rf, n_estimators=n_estimators,random_state=42,verbose=2,max_iter=n_estimators,alpha=0.05)
        
        # Fit Boruta
        boruta_selector.fit(X.values, y)

        # Transform dataframe to include only selected features
        selected_features = X.columns[boruta_selector.support_].tolist()

        # Combine selected features with the columns to retain
        final_columns = selected_features + columns_to_retain

        # Keep only the selected features along with 'is_dominant' and 'patient'
        bdf_reduced = bdf[final_columns]

        return bdf_reduced

        # Return the reduced dataframe with the selected features and the target column

    def rename_columns_with_orientation(self, bdf):
        new_columns = []
        count_dict = {}

        for col in bdf.columns:
            if col not in ['is_dominant', 'patient', 'cohort']:  # Ignore non-feature columns
                base_name, orientation = col.rsplit('_', 1)  # Split into feature name and orientation

                if base_name not in count_dict:
                    count_dict[base_name] = {'x': 0, 'y': 0, 'z': 0}

                new_name = f"{base_name}_{count_dict[base_name][orientation]}_{orientation}"
                count_dict[base_name][orientation] += 1
                new_columns.append(new_name)
            else:
                new_columns.append(col)  # Keep these columns unchanged

        bdf.columns = new_columns
        return bdf




    def optimal_silhouetteb_scores(self, bdf, max_features=500, p_range=False):
        optimal_num_features = Predictor.evaluate_feature_scores(bdf, max_features)

        # If p_range is True, plot the silhouette scores up to the optimal number of features
        if p_range:
            feature_scores = Predictor.evaluate_feature_scores(bdf, optimal_num_features)
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, optimal_num_features + 1), list(feature_scores.values()), marker='o')
            plt.title('Silhouette Scores for Different Numbers of Features')
            plt.xlabel('Number of Features')
            plt.ylabel('Silhouette Score')
            plt.grid(True)
            plt.show()
        else:
            print("Optimal number of features based on silhouette scores:", optimal_num_features)

        return optimal_num_features

# Remove this after new data (add_other)
    def get_df(self, force_load=True, add_other=True):
        def gen_df():
            sensor = Sensor.get(self.sensor_id)
            same_sensor_objs = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)
            x = self.generate_dataframes(sensor=same_sensor_objs[0], add_other=add_other) # X
            print("YOOOO")
            print(x)
            
            is_dom = x['is_dominant']
            pt =x['patient']

            x.columns = [str(col) + '_x' for col in x.columns]
            y = self.generate_dataframes(sensor=same_sensor_objs[1], add_other=add_other) # Y
            y.columns = [str(col) + '_y' for col in y.columns]
            z = self.generate_dataframes(sensor=same_sensor_objs[2], add_other=add_other) # Z
            z.columns = [str(col) + '_z' for col in z.columns]
            columns_to_drop = ['is_dominant_x', 'is_dominant_y', 'is_dominant_z','patient_x', 'patient_y','patient_z', 'cohort_x', 'cohort_y', 'cohort_z']

            # Step 1: Normalize the cohort, is_dominant, and patient columns in original x, y, z before renaming
            bdf = pd.concat([x, y, z], axis=1)

            x_normalized = x[['cohort_x', 'is_dominant_x', 'patient_x']].rename(columns={'cohort_x': 'cohort', 'is_dominant_x': 'is_dominant', 'patient_x': 'patient'}).drop_duplicates()
            y_normalized = y[['cohort_y', 'is_dominant_y', 'patient_y']].rename(columns={'cohort_y': 'cohort', 'is_dominant_y': 'is_dominant', 'patient_y': 'patient'}).drop_duplicates()
            z_normalized = z[['cohort_z', 'is_dominant_z', 'patient_z']].rename(columns={'cohort_z': 'cohort', 'is_dominant_z': 'is_dominant', 'patient_z': 'patient'}).drop_duplicates()
            master_info = pd.concat([x_normalized, y_normalized, z_normalized]).drop_duplicates()
            
            is_dom = bdf['is_dominant_z']
            pt = bdf['patient_z']
            cohort = bdf['cohort_z']

            bdf['is_dominant'] = is_dom
            bdf['patient'] = pt
            bdf['cohort'] = cohort

            bdf = bdf.merge(master_info, on=['is_dominant','patient','cohort'], how='left')
            bdf = bdf.drop(columns=columns_to_drop)

            return bdf

        if self.non_norm == 1:
            # self.update(aggregated_stats_non_normed = self.aggregated_stats)
            if force_load == True:
                self.save()
            if self.aggregated_stats_non_normed is None or force_load == True:
                print("generating non normed agg df...")
                stats = memoryview(pickle.dumps(gen_df()))
                self.update(aggregated_stats_non_normed = stats)
            return pickle.loads(self.aggregated_stats_non_normed)
        
        if self.aggregated_stats is None or force_load == True:
            print("generating agg df...")
            stats = memoryview(pickle.dumps(gen_df()))
            self.update(aggregated_stats = stats)

        return pickle.loads(self.aggregated_stats)


    def get_bdf(self, holdout=False):
        if self.non_norm == 1 or self.non_norm == True:
            bdf = self.get_aggregated_stats_non_normed()
        else:
            bdf = self.get_aggregated_stats()

        if holdout:
            # Assuming each patient has two rows
            self.holdout_patients = bdf['patient'].sample(2)
            self.holdout_data = bdf[bdf['patient'].isin(self.holdout_patients)]
            bdf = bdf[~bdf['patient'].isin(self.holdout_patients)]
        return bdf.copy()


    def get_y(self, bdf):
        return LabelEncoder().fit_transform(bdf['is_dominant'])
    
    

    def clean_bdf(self, bdf):
        def get_patient_id(p_name):
            # Assuming Patient.find_by returns a Patient object if found, None otherwise
            print(p_name)
            return Patient.where(name=p_name)[0].id
    
        bdf.dropna(axis=1, how='all', inplace=True)  # This will drop columns where all values are NaN
        label_encoder = LabelEncoder()
        import pdb;pdb.set_trace()
        try:
            bdf['patient'] = label_encoder.fit_transform(bdf.get('patient', pd.Series()))
        except TypeError:
            # If a TypeError is encountered, proceed with custom conversion
            # Store original patient IDs in a new column
            bdf['str_patient'] = bdf['patient']

            
            # Convert 'patient' column to integers based on the last two digits
            # Extracting last two digits and converting to integer
            patient_ids = []
    
            for p_name in bdf['patient']:
        
                patient_id = get_patient_id(p_name)
                patient_ids.append(patient_id)
            bdf['patient'] = patient_ids


            
            # Move 'str_patient' column in front of 'is_dominant'
            col_order = bdf.columns.tolist()
            # Assuming 'is_dominant' is not the first column, adjust if necessary
            is_dominant_index = col_order.index('is_dominant')
            # Reorder columns to move 'str_patient' in front of 'is_dominant'
            new_order = col_order[:is_dominant_index] + ['str_patient'] + col_order[is_dominant_index:-1]
            bdf = bdf[new_order]

        
        bdf.fillna(0, inplace=True)  # This will replace any remaining NaN values with 0
        
        if np.isinf(bdf).any().any():
            print("Infinities found in DataFrame")
            # Replace infinities with a large number or NaN
            bdf.replace([np.inf, -np.inf], np.nan, inplace=True)
            bdf.dropna(axis=1, how='all', inplace=True)  # This will drop columns where all values are NaN
            bdf.fillna(0, inplace=True)  # This will replace any remaining NaN values with 0
            # Alternatively, you can drop rows with infinite values:
            # bdf = bdf.replace([np.inf, -np.inf], np.nan).dropna(how="any")
        # Check for values too large for float32
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        if bdf.max().max() > float32_max or bdf.min().min() < float32_min:
            print("Values in DataFrame are too large for float32")
        if bdf.isna().any().any():
            print("NaN values found in DataFrame")
        return bdf

    # The method starts considering feature sets from the minimum number of features.
    # It updates the best number of features as long as the silhouette score is improving and above the minimum threshold.
    # It stops once the silhouette score falls below the threshold, given that it has already found a suitable number of features above the threshold.
    # If it doesn't find any feature set above the threshold, it defaults to the minimum number of features.
    @classmethod
    def evaluate_feature_scores(cls, bdf, max_features=500, min_features=50, min_avg_silhouette_score=0.5):
        columns_to_retain = ['is_dominant', 'patient']

        # Remove constant features
        non_constant_features = bdf.loc[:, bdf.nunique() > 1]
        features = non_constant_features.drop(columns=columns_to_retain)
        target = bdf['is_dominant']

        best_num_features = None
        recent_scores = []

        # Start from min_features
        for num_features in range(min_features, max_features + 1):
            selector = SelectKBest(score_func=f_classif, k=num_features)
            selected_features = selector.fit_transform(features, target)

            # Clustering
            clusterer = KMeans(n_clusters=2)
            cluster_labels = clusterer.fit_predict(selected_features)

            # Silhouette Score
            score = silhouette_score(selected_features, cluster_labels)
            recent_scores.append(score)
            if len(recent_scores) > 10:
                recent_scores.pop(0)

            # Check the average of the last 10 scores
            if len(recent_scores) == 10 and np.mean(recent_scores) < min_avg_silhouette_score:
                break
            else:
                best_num_features = num_features

        # If no feature set met the silhouette threshold, fall back to min_features
        return best_num_features if best_num_features is not None else min_features



    def trim_bdf(self, bdf, min_features=10, custom_limit=None):
        bdf_copy = bdf.copy()
        if custom_limit is None:
            optimal_num_features = self.evaluate_feature_scores(bdf_copy, min_features=min_features)
        else:
            optimal_num_features = custom_limit

        # Columns to retain regardless of feature selection
        columns_to_retain = ['is_dominant', 'patient']

        # Separate features and target
        features = bdf.drop(columns=columns_to_retain)
        target = bdf['is_dominant']

        # Apply SelectKBest to select the optimal number of features
        print("Selecting K Best:", optimal_num_features)
        selector = SelectKBest(score_func=f_classif, k=optimal_num_features)
        selector.fit(features, target)
        selected_features = features.columns[selector.get_support()]

        # Combine selected features with the columns to retain
        final_columns = list(selected_features) + columns_to_retain

        # Keep only the selected features along with 'is_dominant' and 'patient'
        bdf_reduced = bdf[final_columns]

        return bdf_reduced

    @classmethod
    def count_scores_above_threshold(cls, scores_dict, threshold=0.5):
        """
        Count the number of silhouette scores above a given threshold.

        Parameters:
        scores_dict (dict): Dictionary with feature counts as keys and silhouette scores as values.
        threshold (float): The threshold value for silhouette scores.

        Returns:
        int: The count of scores above the threshold.
        """
        count = sum(score > threshold for score in scores_dict.values())
        return count
    
    @classmethod
    def only_evaluate_feature_scores(cls, bdf, max_features=50):
        # Assuming bdf is already imputed
        columns_to_retain = ['is_dominant', 'patient']
        features = bdf.drop(columns=columns_to_retain)
        target = bdf['is_dominant']

        feature_scores = {}

        for num_features in range(1, max_features + 1):
            selector = SelectKBest(score_func=f_classif, k=num_features)
            try:
                selected_features = selector.fit_transform(features, target)
            except ValueError as e:
                print(f"Error with {num_features} features: {e}")
                continue

            clusterer = KMeans(n_clusters=2)
            cluster_labels = clusterer.fit_predict(selected_features)

            score = silhouette_score(selected_features, cluster_labels)
            feature_scores[num_features] = score

        return feature_scores

    @classmethod
    def only_plot_silhouette_scores(cls, bdf, max_features):
        scores_dict = cls.only_evaluate_feature_scores(bdf, max_features)
        plt.figure(figsize=(10, 6))
        # Plot the values of the dictionary
        plt.plot(range(1, len(scores_dict) + 1), list(scores_dict.values()), marker='o')
        plt.title('Silhouette Scores vs Number of Features')
        plt.xlabel('Number of Features')
        plt.ylabel('Silhouette Score')
        plt.grid(True)
        plt.show()


    @classmethod
    def get_drop_cols(cls, bdf, threshold):
        corr_matrix = bdf.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop


    def clean_data(self, df):
        """
        Cleans the 'is_dominant', 'patient' columns and other necessary preprocessing.
        
        Parameters:
            df (pd.DataFrame): The dataframe with the columns to clean.
        
        Returns:
            pd.DataFrame: The cleaned dataframe.
        """
        
        # Extract the integer part of the patient ID and convert to integer type
        df['patient'] = df['patient'].str.extract('(\d+)')[0].astype(int)
        
        # Replace NaN values with 0
        df.fillna(0, inplace=True)
        
        # Replace infinities and check for values too large for float32
        if np.isinf(df).any().any():
                        # Replace infinities with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Replace NaN values with 0
            df.fillna(0, inplace=True)
        
        float32_max = np.finfo(np.float32).max
        float32_min = np.finfo(np.float32).min
        if df.max().max() > float32_max or df.min().min() < float32_min:
            print("Values in DataFrame are too large for float32")
        
        if df.isna().any().any():
            print("NaN values found in DataFrame")
        
        return df

    @classmethod
    def median_confusion_matrix(cls, confusion_matrices, accuracy_scores):
        print(confusion_matrices)
        print(accuracy_scores)
        median_accuracy = np.median(accuracy_scores)
        median_index = np.argmin([abs(score - median_accuracy) for score in accuracy_scores])
        median_matrix = confusion_matrices[median_index]
        
        return median_matrix

    def fit_multi_models(self, bdf, y, use_shap=False):
        print(bdf)
        print("SENSOR ID:", self.sensor_id)
        curr_len = len(bdf.index)
        if curr_len < MINIMUM_SAMPLE_SIZE:
            classifiers = MINI_PARAMS
        else:
            width = len(bdf.columns) - 2
            classifiers = self.define_classifiers(feature_count=width) 
        groups = bdf['patient']
        scores = {}
        params = {}
        acc_metrics = {}

        classifier_accuracies, classifier_params, classifier_metrics = {}, {}, {}

        for classifier_name, classifier_data in classifiers.items():
            print(classifier_name, classifier_data)
            print(f"Training {classifier_name}...")
            average_acc, best_params, extra_metrics, scores, params, acc_metrics = self._train_classifier(bdf, y, groups, classifier_name, classifier_data, scores, params, acc_metrics, use_shap=use_shap)
            classifier_accuracies[classifier_name] = np.mean(average_acc)
            print(classifier_name, "Sensor ID:", self.sensor_id, "AVERAGE ACC", np.mean(average_acc))
            classifier_params[classifier_name] = best_params
            classifier_metrics[classifier_name] = extra_metrics

        print("Accuracy Scores:")
        for classifier_name, score in scores.items():
            print(f"{classifier_name}: {score}")
            classifier_accuracies[classifier_name] = score  # Store the scores in the new dictionary
        
        for classifier_name, param in params.items():
            print(f"{classifier_name}: {params}")
            classifier_params[classifier_name] = param  # Store the scores in the new dictionary
        
        for classifier_name, n_metrics in acc_metrics.items():
            print(f"{classifier_name}: {n_metrics}")
            classifier_metrics[classifier_name] = n_metrics 

        self._update_accuracies(classifier_accuracies, classifier_params, classifier_metrics)
        print("done......")
        self.save()

    def is_supported_by_tree_explainer(self, classifier_name):
        return classifier_name in SHAP_CLASSIFIERS

    def neighbor_params(self):
        curr_len = len(self.get_bdf().index)
        if curr_len < MINIMUM_SAMPLE_SIZE:
            return range(1, curr_len - 1)
        return range(5, 15)

    def define_classifiers(self, use_shap_compatible=False, feature_count=50):
        classifiers = {
            'KNN': {
                'classifier': KNeighborsClassifier(),
                'param_grid': {
                    'classifier__n_neighbors': range(1, 11),  # Example range, adjust based on sqrt(n_samples)
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'RandomForest': {
                'classifier': RandomForestClassifier(),
                'param_grid': {
                    # More trees for more complex models, but start lower for initial runs
                    'classifier__n_estimators': [6, 12, 24, 50, 100, 200] if feature_count > 52 else [6, 12, 24, 75],
                    # None allows trees to grow as much as needed, but we limit depth for smaller feature sets
                    'classifier__max_depth': [None, 10, 20, 30] if feature_count > 52 else [None, 2, 4, 8, 16],
                    # Adjusting for the dataset's feature count
                    'classifier__min_samples_split': [2, 4, 6, 10, 15],
                    'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
                    # 'auto' lets the model decide, but specifying can help with large feature sets
                    'classifier__max_features': ['sqrt', 'log2']
                }
            },
            'ExtraTrees': {
                'classifier': ExtraTreesClassifier(),
                'param_grid': {
                    'classifier__n_estimators': [6, 12, 24, 50, 100, 200] if feature_count > 52 else [6, 12, 24, 75],
                    'classifier__max_depth': [None, 20, 40, 80] if feature_count > 52 else [None, 4, 16, 32],
                    'classifier__min_samples_split': [2, 4, 6, 10, 15],
                    'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
                    'classifier__max_features': ['sqrt', 'log2']
                }
            },
            'GradientBoosting': {
                'classifier': GradientBoostingClassifier(),
                'param_grid': {
                    'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7, 9] if feature_count > 52 else [3, 4, 5],
                }
            },
            'DecisionTree': {
                'classifier': DecisionTreeClassifier(),
                'param_grid': {
                    'classifier__max_depth': list(range(3, 14, 2)) if feature_count > 52 else list(range(2, 10)),
                    'classifier__min_samples_split': list(range(2, 11, 2)),
                    'classifier__min_samples_leaf': list(range(1, 5)),
                    'classifier__max_features': ['sqrt', 'log2']
                }
            },
            'LogisticRegression': {
                'classifier': LogisticRegression(max_iter=10000, solver='liblinear'),
                'param_grid': {
                    'classifier__C': np.logspace(-4, 4, 5).tolist(),
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']  # Adjust if considering 'saga' for high-dimensionality
                }
            },
            'XGBoost': {
                'classifier': XGBClassifier(),
                'param_grid': {
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 6, 9],
                    'classifier__subsample': [0.7, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.7, 0.9, 1.0]
                }
            },
            'CatBoost': {
                'classifier': CatBoostClassifier(verbose=0),
                'param_grid': {
                    'classifier__learning_rate': [0.01, 0.05, 0.1],
                    'classifier__iterations': [500, 1000],
                    'classifier__depth': [4, 6, 8]
                }
            },
            'SVM': {
                'classifier': SVC(probability=True),
                'param_grid': {
                    'classifier__C': [0.01, 0.1],  # Reduced and focused range
                    'classifier__gamma': [0.1, 'scale'],  # Reduced range, removed 'auto' and very low value
                    'classifier__kernel': ['linear', 'rbf']  # Focusing on commonly used kernels
                }
            },
            'AdaBoost': {
                'classifier': AdaBoostClassifier(),
                'param_grid': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 1.0]
                }
            },
        }
        
        if use_shap_compatible:
            classifiers = {k: v for k, v in classifiers.items() if self.is_supported_by_tree_explainer(k)}

        return classifiers

    @classmethod
    def define_classifiers_cls(cls, use_shap_compatible, use_mini_params, classifier_name=None, feature_count=50):
        if use_mini_params:
            classifiers = cls.MINI_PARAMS
        else:         
            classifiers = {
                'KNN': {
                    'classifier': KNeighborsClassifier(),
                    'param_grid': {
                        'classifier__n_neighbors': range(1, 11),  # Example range, adjust based on sqrt(n_samples)
                        'classifier__weights': ['uniform', 'distance'],
                        'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
                    }
                },
                'RandomForest': {
                    'classifier': RandomForestClassifier(),
                    'param_grid': {
                        # More trees for more complex models, but start lower for initial runs
                        'classifier__n_estimators': [6, 12, 24, 50, 100, 200] if feature_count > 52 else [6, 12, 24, 75],
                        # None allows trees to grow as much as needed, but we limit depth for smaller feature sets
                        'classifier__max_depth': [None, 10, 20, 30] if feature_count > 52 else [None, 2, 4, 8, 16],
                        # Adjusting for the dataset's feature count
                        'classifier__min_samples_split': [2, 4, 6, 10, 15],
                        'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
                        # 'auto' lets the model decide, but specifying can help with large feature sets
                        'classifier__max_features': ['sqrt', 'log2']
                    }
                },
                'ExtraTrees': {
                    'classifier': ExtraTreesClassifier(),
                    'param_grid': {
                        'classifier__n_estimators': [6, 12, 24, 50, 100, 200] if feature_count > 52 else [6, 12, 24, 75],
                        'classifier__max_depth': [None, 20, 40, 80] if feature_count > 52 else [None, 4, 16, 32],
                        'classifier__min_samples_split': [2, 4, 6, 10, 15],
                        'classifier__min_samples_leaf': [1, 2, 4, 6, 10],
                        'classifier__max_features': ['sqrt', 'log2']
                    }
                },
                'GradientBoosting': {
                    'classifier': GradientBoostingClassifier(),
                    'param_grid': {
                        'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
                        'classifier__n_estimators': [100, 200, 300],
                        'classifier__max_depth': [3, 5, 7, 9] if feature_count > 52 else [3, 4, 5],
                    }
                },
                'DecisionTree': {
                    'classifier': DecisionTreeClassifier(),
                    'param_grid': {
                        'classifier__max_depth': list(range(3, 14, 2)) if feature_count > 52 else list(range(2, 10)),
                        'classifier__min_samples_split': list(range(2, 11, 2)),
                        'classifier__min_samples_leaf': list(range(1, 5)),
                        'classifier__max_features': ['sqrt', 'log2', None] if feature_count > 52 else ['log2'],
                    }
                },
                'LogisticRegression': {
                    'classifier': LogisticRegression(max_iter=10000, solver='liblinear'),
                    'param_grid': {
                        'classifier__C': np.logspace(-4, 4, 5).tolist(),
                        'classifier__penalty': ['l1', 'l2'],
                        'classifier__solver': ['liblinear']  # Adjust if considering 'saga' for high-dimensionality
                    }
                },
                'XGBoost': {
                    'classifier': XGBClassifier(),
                    'param_grid': {
                        'classifier__learning_rate': [0.01, 0.1, 0.2],
                        'classifier__n_estimators': [100, 200, 300],
                        'classifier__max_depth': [3, 6, 9],
                        'classifier__subsample': [0.7, 0.9, 1.0],
                        'classifier__colsample_bytree': [0.7, 0.9, 1.0]
                    }
                },
                'CatBoost': {
                    'classifier': CatBoostClassifier(verbose=0),
                    'param_grid': {
                        'classifier__learning_rate': [0.01, 0.05, 0.1],
                        'classifier__iterations': [500, 1000],
                        'classifier__depth': [4, 6, 8]
                    }
                },
                'SVM': {
                    'classifier': SVC(probability=True),
                    'param_grid': {
                        'classifier__C': [0.01, 0.1],  # Reduced and focused range
                        'classifier__gamma': [0.1, 'scale'],  # Reduced range, removed 'auto' and very low value
                        'classifier__kernel': ['linear', 'rbf']  # Focusing on commonly used kernels
                    }
                },
                'AdaBoost': {
                    'classifier': AdaBoostClassifier(),
                    'param_grid': {
                        'classifier__n_estimators': [50, 100, 200],
                        'classifier__learning_rate': [0.01, 0.1, 1.0]
                    }
                },
            }
        def in_te(classifier_name):
            classifier_name in SHAP_CLASSIFIERS

        if use_shap_compatible:
            classifiers = {k: v for k, v in classifiers.items() if in_te(k)}

        if classifier_name is not None:
            return classifiers.get(classifier_name, None)
        
        return classifiers

    @classmethod
    def _fix_odd_test_set(cls, X_train, y_train, X_test, y_test):
        """Ensure that the test set has an even size."""
        if len(X_test) % 2 != 0:

            # Count the occurrence of each patient in the test set
            patient_counts = Counter(X_test['patient'])
            # Find the patient that occurs only once
            single_occurrence_patient = [patient for patient, count in patient_counts.items() if count == 1]
                        
            if single_occurrence_patient:
                # Find the index of the single occurrence patient in the test set
                single_patient_index = X_test[X_test['patient'] == single_occurrence_patient[0]].index[0]

                # Move this patient from test set to train set
                X_train = X_train.append(X_test.loc[single_patient_index])
                y_train = y_train.append(pd.Series(y_test.loc[single_patient_index]))
                            
                # Drop the patient from test set
                X_test = X_test.drop(single_patient_index)
                y_test = y_test.drop(single_patient_index)
                            
                # Reset the index for the updated DataFrames
                X_train = X_train.reset_index(drop=True)
                y_train = y_train.reset_index(drop=True)
                X_test = X_test.reset_index(drop=True)
                y_test = y_test.reset_index(drop=True)

        return X_train, y_train, X_test, y_test


    def _train_classifier(self, bdf, y, groups, classifier_name, classifier_data, scores, params, acc_metrics, use_shap=False):
        # Training of a specific classifier

        classifier = classifier_data['classifier']
        pipe = Pipeline(steps=[('classifier', classifier)])
        param_grid_classifier = classifier_data['param_grid']
        cross_val_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
        f1_scores, auc_roc_scores, log_loss_scores, confusion_matrices, all_importance_dfs = [], [], [], [], []

        all_X = []
        combined_shap_values = []

        if len(bdf.index) < MINIMUM_SAMPLE_SIZE:
            splitter = LeaveOneOut()
            split = splitter.split(bdf, y)
            round_set_size = False
        else:
            splitter = StratifiedKFold(n_splits=DEFAULT_K_FOLD_SPLITS)
            split = splitter.split(bdf, y, groups)
            round_set_size = True

        for train_index, test_index in split:
            X_train, X_test = bdf.iloc[train_index], bdf.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Convert y_train and y_test to pandas Series
            y_train, y_test = pd.Series(y_train, index=train_index), pd.Series(y_test, index=test_index)

            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            if round_set_size:
                X_train, y_train, X_test, y_test = self._fix_odd_test_set(X_train, y_train, X_test, y_test)
                
            # Remove 'patient' column if it exists in the dataframe
            if 'patient' in X_train.columns:
                X_train = X_train.drop('patient', axis=1)
                X_test = X_test.drop('patient', axis=1)

            # Remove 'is_dominant' column
            X_train = X_train.drop('is_dominant', axis=1)
            X_test = X_test.drop('is_dominant', axis=1)

            grid_search = GridSearchCV(pipe, param_grid_classifier, cv=splitter, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
                    # Store best parameters and score
            train_accuracy = grid_search.score(X_train, y_train)
            best_classifier = grid_search.best_estimator_.named_steps['classifier']
            current_best_score = grid_search.best_score_
            current_best_params = grid_search.best_params_

            print(
                "Training accuracy...", train_accuracy, 
                "Current best training score:", current_best_score, 
                "Current best training params:", current_best_params,
                "Current best training classifier", best_classifier
            )
            # Predict on test set and calculate metrics
            y_pred = best_classifier.predict(X_test)
            y_pred_proba = best_classifier.predict_proba(X_test)[:, 1] if hasattr(best_classifier, "predict_proba") else None
    
            # Check if the model is compatible with SHAP TreeExplainer
            if use_shap is True and best_classifier.__class__.__name__ in COMPATIBLE_MODELS:
                print("Using SHAP!")
                explainer = shap.TreeExplainer(best_classifier)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):  # If there are multiple classes
                    # Taking only the SHAP values for class 1 in case of binary classification.
                    # Adjust for multi-class problems as necessary.
                    shap_values = shap_values[1]
                combined_shap_values.append(shap_values)
                all_X.append(X_test)

            if hasattr(classifier, 'feature_importances_'):
                feature_importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': feature_importances
                })

                # Append this dataframe to the list
                all_importance_dfs.append(importance_df)
            else:
                print(f"{type(classifier).__name__} doesn't have feature_importances_ attribute.")

            accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
            precision_scores.append(metrics.precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(metrics.recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(metrics.f1_score(y_test, y_pred, average='weighted'))
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
            cross_val_scores.append(current_best_score)
            if len(np.unique(y_test)) > 1:
                auc_roc_scores.append(metrics.roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else np.nan)
                log_loss_scores.append(metrics.log_loss(y_test, y_pred_proba) if y_pred_proba is not None else np.nan)

            # Compute metrics

        if hasattr(classifier, 'feature_importances_'):
            # After the loop over the K-Folds is completed (outside the loop):
            # Concatenate all importance dataframes
            all_importance_concat = pd.concat(all_importance_dfs)
            # Group by Feature and compute median
            median_importance = all_importance_concat.groupby('Feature').median().reset_index()

            # Get the top 5 features with highest median importance
            top_5_median_features_df = median_importance.nlargest(5, 'Importance')
            # Convert the dataframe to a dictionary for storage
            top_5_median_features = dict(zip(top_5_median_features_df['Feature'], top_5_median_features_df['Importance']))
            print(top_5_median_features)

        else: 
            top_5_median_features = {}

        # After the cross-validation loop but before returning
        if hasattr(self, 'holdout_data') and self.holdout_data is not None:
            # Ensure Consistent Features Across Datasets
            holdout_features = self.holdout_data.drop('is_dominant', axis=1).columns.tolist()  # dropping the label column
            bdf_features = bdf.columns.tolist()

            if set(holdout_features) != set(bdf_features):
                missing_in_holdout = set(bdf_features) - set(holdout_features)
                extra_in_holdout = set(holdout_features) - set(bdf_features)
                
                if missing_in_holdout:
                    print(f"Warning: The holdout data is missing the following columns compared to training data: {missing_in_holdout}")
                if extra_in_holdout:
                    print(f"Warning: The holdout data has additional columns not present in the training data: {extra_in_holdout}")

            # Extract labels from holdout_data
            y_holdout = self.get_y(self.holdout_data)
            
            # Extract features from holdout_data (excluding the 'is_dominant' column)
            X_holdout = self.holdout_data.copy()
            X_holdout = X_holdout[X_train.columns]
            X_holdout = self.clean_data(X_holdout)


            # Predictions using the best trained model from grid_search
            y_holdout_pred = grid_search.predict(X_holdout)

            # Compute metrics for holdout set
            holdout_accuracy = metrics.accuracy_score(y_holdout, y_holdout_pred)
            holdout_precision = metrics.precision_score(y_holdout, y_holdout_pred, average='weighted')
            holdout_recall = metrics.recall_score(y_holdout, y_holdout_pred, average='weighted')
            holdout_f1 = metrics.f1_score(y_holdout, y_holdout_pred, average='weighted')

            print("Holdout set metrics:")
            print("Accuracy:", holdout_accuracy)
            print("Precision:", holdout_precision)
            print("Recall:", holdout_recall)
            print("F1-score:", holdout_f1)

        if use_shap is True and best_classifier.__class__.__name__ in COMPATIBLE_MODELS:
            print("Using SHAP 2!")
            aggregated_shap_values = np.concatenate(combined_shap_values, axis=0)
            combined_X = pd.concat(all_X, axis=0).reset_index(drop=True)

            cpc = PredictorScore.where(classifier_name=classifier_name, predictor_id=self.id, score_type="ShapleySummary")
            if len(cpc) == 0:
                cpc = PredictorScore.find_or_create(classifier_name=classifier_name,score_type="ShapleySummary",predictor_id=self.id)
            else:
                cpc = cpc[0]
            cpc.set_shap_matrix(aggregated_shap_values=aggregated_shap_values, combined_X=combined_X)
        
        average_training_score = np.mean(cross_val_scores)
        average_accuracy = np.mean(accuracy_scores)
        average_precision = np.mean(precision_scores)
        average_recall = np.mean(recall_scores)
        average_f1 = np.mean(f1_scores)
        average_auc_roc = np.nanmean(auc_roc_scores)  # Handle potential NaNs in AUC-ROC
        average_log_loss = np.nanmean(log_loss_scores)  # Handle potential NaNs in log loss

        print("Training Accuracy...", average_training_score, "Test Accuracy...", accuracy_scores)

    # Prepare and return metrics and other information
        extra_metrics = {
            "Important Features": top_5_median_features,
            "Training Score": average_training_score,
            "Accuracy": average_accuracy,
            "Precision": average_precision,
            "Recall": average_recall,
            "F1-score": average_f1,
            "AUC-ROC": average_auc_roc,
            "Log loss": average_log_loss,
            "Median Confusion Matrix": Predictor.median_confusion_matrix(confusion_matrices, accuracy_scores).tolist(),
            "Skip Boruta": str(self.skip_boruta)
        }

        scores[classifier_name] = average_accuracy
        params[classifier_name] = current_best_params
        acc_metrics[classifier_name] = extra_metrics

        return average_accuracy, grid_search.best_params_, extra_metrics, scores, params, acc_metrics

    # Helper functions for _train_classifier
    # Including:
    # - _split_data: To split data into training and test sets
    # - _fix_odd_test_set: To fix the test set if it has an odd size
    # - _compute_metrics: To compute various metrics
    # - _calculate_average_metrics: To calculate average metrics

    def _update_accuracies(self, classifier_accuracies, classifier_params, classifier_metrics):
        # Update or add classifier accuracies, parameters, and metrics
        self.accuracies = self.multiple_deserialize(self.accuracies) or {}
        self.accuracies['classifier_accuracies'] = self.accuracies.get('classifier_accuracies', {})
        self.accuracies['classifier_params'] = self.accuracies.get('classifier_params', {})
        self.accuracies['classifier_metrics'] = self.accuracies.get('classifier_metrics', {})

        self.accuracies['classifier_accuracies'].update(classifier_accuracies)
        self.accuracies['classifier_params'].update(classifier_params)
        self.accuracies['classifier_metrics'].update(classifier_metrics)

        self._print_results(classifier_accuracies, classifier_params, classifier_metrics)

    def _print_results(self, accuracies, params, metrics):
        # Print classifier results
        print("Accuracy Scores:")
        for name, score in accuracies.items():
            print(f"{name}: {score}")

        for name, param in params.items():
            print(f"{name}: {params}")

        for name, n_metrics in metrics.items():
            print(f"{name}: {n_metrics}")


    def multiple_deserialize(self, serialized_string):
        while isinstance(serialized_string, str):
            try:
                serialized_string = json.loads(serialized_string)
            except json.JSONDecodeError:
                break
        return serialized_string

    def save(self):
        """
        Save the current object state to the database.
        """
        # matrix_json = json.dumps(self.matrix)
        json_accuracies = json.dumps(self.accuracies)
        if self.matrix is not None:
            mat = memoryview(self.matrix)
        else:
            mat = None

        # update the object properties in database
        updated_rows = self.update(
            id=self.id,
            task_id=self.task_id,
            sensor_id=self.sensor_id,
            non_norm=self.non_norm,
            abs_val=self.abs_val,
            matrix=mat,
            accuracies=json_accuracies
        )
    
        # You might need a commit after updating the rows. It will depend on your specific DB library
        self.__class__._conn.commit()
        return updated_rows > 0
    
    def get_classifier_accuracies(self, alt=None):
        new_acc_dict = {}
        metrics = self.get_accuracies()['classifier_metrics']
        print(metrics)
        print(alt)
        for model_type in self.get_accuracies()['classifier_accuracies'].keys():
            # Check if "Accuracy:" is in the metrics, otherwise fall back to "Accuracy"
            
            if "Accuracy:" in metrics[model_type]:
                accuracy_key = "Accuracy:" 
            else:
                accuracy_key = "Accuracy"
            
            if alt == "training":
                accuracy_key = "Training Score"
            
            # Use the correct key to get the accuracy
            print(metrics[model_type])
            print()
            print(self.attrs())
            new_acc_dict[model_type] = metrics[model_type][accuracy_key]

        return new_acc_dict

    def get_predictor_scores(self, score_type=None, model_name=None):
        scores = None
        if score_type is not None and model_name is not None:
            scores = PredictorScore.where(predictor_id=self.id, score_type=score_type, model_name=model_name)
        elif score_type is not None:
            scores = PredictorScore.where(predictor_id=self.id, score_type=score_type)
        elif model_name is not None:
            scores = PredictorScore.where(predictor_id=self.id, classifier_name=model_name)
        else:
            scores = PredictorScore.where(predictor_id=self.id)
        
        if scores is None or scores is []:
            return [None]
        return scores


    def get_predictor_scores_by_classifier_names(self, classifier_names):
        # Query the predictor_score table where predictor_id matches this instance's id
        return PredictorScore.where(predictor_id=self.id, classifier_name=classifier_names)
    
    def get_accuracies(self):
        return self.multiple_deserialize(self.accuracies)

    def get_metrics(self):
        return self.multiple_deserialize(self.accuracies)['classifier_metrics']

    def get_classifiers(self):
        return self.multiple_deserialize(self.accuracies)['classifier_metrics'].keys()
    
    def get_feature_importance(self):
        feature_importance = {}
        metrics = self.get_metrics()

        for classifier in self.get_classifiers():
            if classifier in metrics and 'Important Features' in metrics[classifier]:
                feature_importance[classifier] = metrics[classifier]['Important Features']
            else:
                print(f"Warning: 'Important Features' not found for classifier {classifier}")
        
        return feature_importance

    @classmethod
    def _train_from_mp(cls, bdf, y, groups, classifier_name, classifier_data, scores, params, acc_metrics, mps, use_shap=True):
        # Training of a specific classifier
        classifier = classifier_data['classifier']
        pipe = Pipeline(steps=[('classifier', classifier)])
        param_grid_classifier = classifier_data['param_grid']
        cross_val_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
        f1_scores, auc_roc_scores, log_loss_scores, confusion_matrices, all_importance_dfs = [], [], [], [], []

        all_X = []
        combined_shap_values = []

        if len(bdf.index) < MINIMUM_SAMPLE_SIZE:
            splitter = LeaveOneOut()
            split = splitter.split(bdf, y)
            round_set_size = False
        else:
            splitter = StratifiedKFold(n_splits=DEFAULT_K_FOLD_SPLITS)
            split = splitter.split(bdf, y, groups)
            round_set_size = True
        
        for train_index, test_index in split:
            X_train, X_test = bdf.iloc[train_index].drop('is_dominant', axis=1), bdf.iloc[test_index].drop('is_dominant', axis=1)
            y_train, y_test = y[train_index], y[test_index]

            # Convert y_train and y_test to pandas Series
            y_train, y_test = pd.Series(y_train, index=train_index), pd.Series(y_test, index=test_index)

            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            if round_set_size:
                X_train, y_train, X_test, y_test = cls._fix_odd_test_set(X_train, y_train, X_test, y_test)
            
            # Remove 'patient' column if it exists in the dataframe
            if 'patient' in X_train.columns:
                X_train = X_train.drop('patient', axis=1)
                X_test = X_test.drop('patient', axis=1)

            grid_search = GridSearchCV(pipe, param_grid_classifier, cv=splitter, n_jobs=-1, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            train_accuracy = grid_search.score(X_train, y_train)
            best_classifier = grid_search.best_estimator_.named_steps['classifier']
            current_best_score = grid_search.best_score_
            current_best_params = grid_search.best_params_

            print(
                "Training accuracy...", train_accuracy, 
                "Current best training score:", current_best_score, 
                "Current best training params:", current_best_params,
                "Current best training classifier", best_classifier
            )
            # Predict on test set and calculate metrics
            y_pred = best_classifier.predict(X_test)
            y_pred_proba = best_classifier.predict_proba(X_test)[:, 1] if hasattr(best_classifier, "predict_proba") else None

            # Check if the model is compatible with SHAP TreeExplainer
            if use_shap is True and best_classifier.__class__.__name__ in COMPATIBLE_MODELS:
                explainer = shap.TreeExplainer(best_classifier)
                shap_values = explainer.shap_values(X_test)
                if isinstance(shap_values, list):  # If there are multiple classes
                    # Taking only the SHAP values for class 1 in case of binary classification.
                    # Adjust for multi-class problems as necessary.
                    shap_values = shap_values[1]
                combined_shap_values.append(shap_values)
                all_X.append(X_test)


            if hasattr(classifier, 'feature_importances_'):
                feature_importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': feature_importances
                })

                # Append this dataframe to the list
                all_importance_dfs.append(importance_df)
            else:
                print(f"{type(classifier).__name__} doesn't have feature_importances_ attribute.")

            # Compute metrics
            confusion_matrices.append(confusion_matrix(y_test, y_pred))

            accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
            print("Test accuracy...", metrics.accuracy_score(y_test, y_pred))
            precision_scores.append(metrics.precision_score(y_test, y_pred, average='weighted'))
            recall_scores.append(metrics.recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(metrics.f1_score(y_test, y_pred, average='weighted'))
            if len(np.unique(y_test)) > 1:
                auc_roc_scores.append(metrics.roc_auc_score(y_test, y_pred_proba))
                log_loss_scores.append(metrics.log_loss(y_test, y_pred_proba))

            cross_val_scores.append(current_best_score)


        if use_shap is True and (classifier_name in SHAP_CLASSIFIERS):
            aggregated_shap_values = np.concatenate(combined_shap_values, axis=0)
            combined_X = pd.concat(all_X, axis=0).reset_index(drop=True)

            cpc = PredictorScore.find_or_create(classifier_name=classifier_name,score_type="MultiShapleySummary", multi_predictor_id=mps.id)
            
            cpc.set_shap_matrix(aggregated_shap_values=aggregated_shap_values, combined_X=combined_X)


        extra_metrics = {}
        # try different value here, interactions of different features
        # may be non linear relationship between features, not sure if this is the case for diff iterations
        # try running this a bunch of times
        if hasattr(classifier, 'feature_importances_'):
            # After the loop over the K-Folds is completed (outside the loop):
            # Concatenate all importance dataframes
            all_importance_concat = pd.concat(all_importance_dfs)
            # Group by Feature and compute median
            median_importance = all_importance_concat.groupby('Feature').median().reset_index()

            # Get the top 5 features with highest median importance
            top_5_median_features_df = median_importance.nlargest(5, 'Importance')
            # Convert the dataframe to a dictionary for storage
            top_5_median_features = dict(zip(top_5_median_features_df['Feature'], top_5_median_features_df['Importance']))
            print(top_5_median_features)

        else: 
            top_5_median_features = {}
    
        test_acc = np.mean(accuracy_scores)
        training_acc = np.mean(cross_val_scores)
        extra_metrics = {
            "Important Features": top_5_median_features,
            "Training Score": training_acc,
            "Accuracy": test_acc,
            "Precision": np.mean(precision_scores),
            "Recall": np.mean(recall_scores),
            "F1-score": np.mean(f1_scores),
            "AUC-ROC": np.mean(auc_roc_scores),
            "Log loss": np.mean(log_loss_scores),
            "Median Confusion Matrix": cls.median_confusion_matrix(confusion_matrices, accuracy_scores).tolist(),
            "Average Training Score": training_acc,
        }

        print("Average Training Accuracy...", training_acc, "Average Test Accuracy...", test_acc)

        scores[classifier_name] = test_acc
        params[classifier_name] = current_best_params
        acc_metrics[classifier_name] = extra_metrics

        return accuracy_scores, grid_search.best_params_, extra_metrics, scores, params, acc_metrics
