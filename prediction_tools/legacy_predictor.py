from collections import Counter
import datetime
import shap
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient_task import PatientTask
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from models.legacy_task import Task
import pandas as pd
import pickle

from models.legacy_sensor import Sensor
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from models.legacy_patient import Patient
import json
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from prediction_tools.predictor_score import PredictorScore
print(tf.__version__)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from prediction_tools.predictor_score import PredictorScore
from sklearn.naive_bayes import GaussianNB  # Assuming features are continuous

from sklearn.model_selection import LeaveOneOut



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
        super().__init__()
        self.id = id
        self.task_id = task_id
        self.sensor_id = sensor_id
        self.non_norm = non_norm
        self.abs_val = abs_val
        self.accuracies = accuracies
        self.matrix = matrix
        self.aggregated_stats_non_normed = aggregated_stats_non_normed
        self.aggregated_stats = aggregated_stats
        self.multi_predictor_id = multi_predictor_id
        self.cohort_id = cohort_id
    
    def sensor(self):
        return Sensor.get(self.sensor_id)
    
    def task(self):
        return Task.get(self.task_id)

    def non_dom_sensor(self, sensor=None):
        if sensor is not None:
            sensor = Sensor.get(sensor.id)
        else:
            sensor = Sensor.get(self.sensor_id)
        task = Task.get(self.task_id)
        nondom_sensor_name = task.get_opposite_sensor(sensor.name)
        nondom = Sensor.find_by("name", nondom_sensor_name)
        return nondom

    def select_features(self):
        sensor = Sensor.get(self.sensor_id)
        nondom_sensor = self.non_dom_sensor()
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

        # self.created_at = self.aggregated_stats_non_normed
        # self.updated_at = self.aggregated_stats
        # self.aggregated_stats_non_normed = self.updated_at
        # self.aggregated_stats = None
        
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

    def train_it(self, force_load=False):
        # very wide grid based on all the features (instead of one from earlier)
        # (combines x, y, z)
        df = self.get_df(force_load=force_load)

        # to get from ~2300 cols wide to ~50
        self.optimize_hyperparameters(df)
        self.save()
        print("done training")
        return self
        

    def get_df(self, force_load=False):
        def gen_df():
            sensor = Sensor.get(self.sensor_id)
            same_sensor_objs = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)
            x = self.generate_dataframes(sensor=same_sensor_objs[0]) # X
            
            is_dom = x['is_dominant']
            pt =x['patient']
            
            x.columns = [str(col) + '_x' for col in x.columns]
            y = self.generate_dataframes(sensor=same_sensor_objs[1]) # Y
            y.columns = [str(col) + '_y' for col in y.columns]
            z = self.generate_dataframes(sensor=same_sensor_objs[2]) # Z
            z.columns = [str(col) + '_z' for col in z.columns]
            columns_to_drop = ['is_dominant_x', 'is_dominant_y', 'is_dominant_z','patient_x', 'patient_y','patient_z']
            

            bdf = pd.concat([x, y, z], axis=1)
            bdf = bdf.drop(columns=columns_to_drop)
            bdf['is_dominant'] = is_dom
            bdf['patient'] = pt
            return bdf
        

        if self.non_norm == 1:
            # self.update(aggregated_stats_non_normed = self.aggregated_stats)
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

    def generate_dataframes(self, sensor=None):
        print("generating df")
        dom_task = Task.get(self.task_id)
        counterpart_task = dom_task.get_counterpart_task()[0]
        if sensor is None:
            sensor = Sensor.get(self.sensor_id)
    
        nondom_sensor = self.non_dom_sensor(sensor=sensor)
        self_pts = PatientTask.where(task_id=dom_task.id, cohort_id=self.cohort_id)

        dom_dataframes = []
        nondom_dataframes = []
        features_dom = self.select_features()[0]
        features_non_dom = self.select_features()[1]

        for self_pt in self_pts:
            # Skip certain patients
            if self_pt.id == 24 or self_pt.id == 29:
                continue
            try:
                counterpart_pts = PatientTask.where(task_id=counterpart_task.id, patient_id=self_pt.patient_id, cohort_id=self.cohort_id)
                counterpart_sensor = nondom_sensor
                if not counterpart_pts:
                    continue
                counterpart_pt = counterpart_pts[0]
                curr_patient = Patient.where(id=self_pt.patient_id)[0].name
                self_temp, counter_temp = self.get_temp_dataframes(self_pt, counterpart_pt, sensor, counterpart_sensor, curr_patient, features_dom, features_non_dom)
                # switch places for lefty patient TODO: something more general here...
                if self_pt.patient_id == 21 or self_pt.patient_id == 26 or self_pt.patient_id == 27 or self_pt.patient_id == 28:
                    print("switching places for lefty pt....")
                    dom_dataframes.append(counter_temp)
                    nondom_dataframes.append(self_temp)
                    print("done....")
                else:
                    dom_dataframes.append(self_temp)
                    nondom_dataframes.append(counter_temp)
            except (TypeError, KeyError) as e:
                print(e)
                
                continue
        dom_df = pd.concat(dom_dataframes)
        nondom_df = pd.concat(nondom_dataframes)
        return pd.concat([dom_df, nondom_df])

    def get_temp_dataframes(self, self_pt, counterpart_pt, sensor, counterpart_sensor, curr_patient, features_dom, features_non_dom):
        features_dom = False
        features_non_dom = False
        
        if self.abs_val is True:
            print("getting temp df abs")
            self_temp = pd.DataFrame(abs(self_pt.combined_gradient_set_stats_list(sensor, abs_val=self.abs_val, non_normed=self.non_norm, loc=features_dom)['mean'])).T
            counter_temp = pd.DataFrame(abs(counterpart_pt.combined_gradient_set_stats_list(counterpart_sensor, abs_val=self.abs_val, non_normed=self.non_norm, loc=features_non_dom)['mean'])).T
        else:
            print("getting temp df norm")
            self_temp = pd.DataFrame(self_pt.combined_gradient_set_stats_list(sensor, abs_val=self.abs_val, non_normed=self.non_norm, loc=features_dom)['mean']).T
            counter_temp = pd.DataFrame(counterpart_pt.combined_gradient_set_stats_list(counterpart_sensor, abs_val=self.abs_val, non_normed=self.non_norm, loc=features_non_dom)['mean']).T
        print("yolo")
        self_temp.columns = ['grad_data__' + col.split('__')[1] for col in self_temp.columns]
        
        # likely overfitting if one patient changes results
        if curr_patient == 'S017':
            self_temp['is_dominant'] = False  # dominant class
        else:
            self_temp['is_dominant'] = True  # dominant class
        self_temp['patient'] = curr_patient



        counter_temp.columns = ['grad_data__' + col.split('__')[1] for col in counter_temp.columns]
        if curr_patient == 'S017':
            counter_temp['is_dominant'] = True
        else:
            counter_temp['is_dominant'] = False  # non-dominant class
        counter_temp['patient'] = curr_patient

        return self_temp, counter_temp

    def rename_duplicate_columns(self, df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique(): 
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df
    

    def get_bdf(self, holdout=False):
        print("yo")
        if self.non_norm == 1:
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

    def trim_bdf(self, bdf=None):
        if bdf is None:
            bdf = self.get_bdf()

        label_encoder = LabelEncoder()
        bdf['patient'] = label_encoder.fit_transform(bdf.get('patient', pd.Series()))
        pt = bdf['patient'].copy()
        bdf = bdf.loc[:, bdf.nunique() != 1]

        
        corr_matrix = bdf.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # If we have very few rows, its easy to see false correlations
        # so far the ones that show up are similar to the prev non cp pt
        # so even if the correlations are way higher, some of the order seems to remain.
        if len(bdf.index) < MINIMUM_SAMPLE_SIZE:
            threshold = .999
        else:
            threshold = .95
         
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        bdf.drop(to_drop, axis=1, inplace=True)
        bdf['patient'] = pt
        return bdf

    def clean_bdf(self, bdf):
        bdf.dropna(axis=1, how='all', inplace=True)  # This will drop columns where all values are NaN
        label_encoder = LabelEncoder()
        bdf['patient'] = label_encoder.fit_transform(bdf.get('patient', pd.Series()))
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
    
    def retrain_from(self, obj=None, use_shap=False, force_load=False):   
        if obj is None:
            obj = self
        # get df
        bdf = self.get_df(force_load=force_load)
        bdf = self.get_bdf(holdout=False)

        y = self.get_y(bdf)
        bdf['is_dominant'] = y
        is_dominant = bdf['is_dominant'].copy()
        bdf = self.clean_bdf(bdf) 
        bdf = self.trim_bdf(bdf)
        bdf['is_dominant'] = is_dominant

        self.fit_multi_models(bdf, y, use_shap)
        print("Done retraining!")

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

    def optimize_hyperparameters(self, dfs=[], best_score=None, best_params=None, n_iter=0):
        bdf = self.get_bdf()
        y = self.get_y(bdf)
        bdf['is_dominant'] = y
        bdf = self.clean_bdf(bdf) 
        bdf = self.trim_bdf(bdf)
        # bdf = bdf.drop('is_dominant', inplace=True, axis=1)

        self.fit_multi_models(bdf, y, True)

    def median_confusion_matrix(self, confusion_matrices, accuracy_scores):
        print(confusion_matrices)
        print(accuracy_scores)
        median_accuracy = np.median(accuracy_scores)
        median_index = np.argmin([abs(score - median_accuracy) for score in accuracy_scores])
        median_matrix = confusion_matrices[median_index]
        
        return median_matrix

    def fit_multi_models(self, bdf, y, use_shap=False):
        curr_len = len(bdf.index)
        mini_params = False
        if curr_len < MINIMUM_SAMPLE_SIZE:
            mini_params = True


        classifiers = self.define_classifiers(use_shap, mini_params) 




        groups = bdf['patient']
        scores = {}
        params = {}
        acc_metrics = {}

        classifier_accuracies, classifier_params, classifier_metrics = {}, {}, {}

        for classifier_name, classifier_data in classifiers.items():
            print(classifier_name, classifier_data)
            print(f"Training {classifier_name}...")
            classifier_scores, best_params, extra_metrics, scores, params, acc_metrics = self._train_classifier(bdf, y, groups, classifier_name, classifier_data, scores, params, acc_metrics, use_shap=use_shap)
            classifier_accuracies[classifier_name] = np.mean(classifier_scores)
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
        # List of classifier names known to be supported by TreeExplainer
        supported_classifiers = [
            'XGBoost', 'RandomForest', 'DecisionTree', 'ExtraTrees', 'CatBoost', 'GradientBoosting'
        ]
        
        return classifier_name in supported_classifiers

    def neighbor_params(self):
        curr_len = len(self.get_bdf().index)
        if curr_len < MINIMUM_SAMPLE_SIZE:
            return range(1, curr_len - 1)
        return range(5, 15)

    def define_classifiers(self, use_shap_compatible, use_mini_params):
        if use_mini_params:
            classifiers = MINI_PARAMS
            if use_shap_compatible:
                classifiers = {k: v for k, v in classifiers.items() if self.is_supported_by_tree_explainer(k)}
            return classifiers
    
        classifiers = {
            'KNN': {
                'classifier': KNeighborsClassifier(),
                'param_grid': {
                    'classifier__n_neighbors': self.neighbor_params(),
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__metric': ['euclidean', 'manhattan', 'minkowski']
                }
            },
            'RandomForest': {
                'classifier': RandomForestClassifier(),
                'param_grid': {
                    'classifier__n_estimators': [30, 50, 75, 125, 175],
                    'classifier__max_depth': [2, 3, 4],
                    'classifier__min_samples_split': [1, 2, 4, 6],
                    'classifier__min_samples_leaf': [1, 2, 3, 4, 5]
                }
            },
            'LogisticRegression': {
                'classifier': LogisticRegression(max_iter=10000, solver='liblinear'),
                'param_grid': {
                    'classifier__C': [.005, 0.01, 0.05, 0.1, 1, 10],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear', 'saga']
                }
            },
            'XGBoost': {
                'classifier': XGBClassifier(),
                'param_grid': {
                    'classifier__learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
                    'classifier__n_estimators': [25, 50, 75, 90, 100],
                    'classifier__max_depth': [1, 2, 3, 4, 5]
                }
            },
            'CatBoost': {
                'classifier': CatBoostClassifier(verbose=0),
                'param_grid': {
                    'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1],
                    'classifier__iterations': [250, 500, 1000],
                    'classifier__depth': [4, 6, 8, 10]
                }
            },
            'SVM': {
                'classifier': SVC(probability=True),
                'param_grid': {
                    'classifier__C': [0.05, 0.1, 1, 10],
                    'classifier__gamma': ['scale', 'auto', 0.1, 1, 10, 100],
                    'classifier__kernel': ['linear', 'poly']
                }
            },
            'AdaBoost': {
                'classifier': AdaBoostClassifier(),
                'param_grid': {
                    'classifier__n_estimators': [30, 50, 75, 100, 125],
                    'classifier__learning_rate': [0.001, 0.01, 0.05, 0.1]
                }
            },
            'ExtraTrees': {
                'classifier': ExtraTreesClassifier(),
                'param_grid': {
                    'classifier__n_estimators': [75, 100, 125, 150, 175],
                    'classifier__max_depth': [3, 4, 6, 8, 10],
                    'classifier__min_samples_split': [2, 4, 6, 8],
                    'classifier__min_samples_leaf': [1, 2, 3, 4, 5]
                }
            },
            'GradientBoosting': {
                'classifier': GradientBoostingClassifier(),
                'param_grid': {
                    'classifier__learning_rate': [.0005, 0.001, 0.01, 0.03, 0.05],
                    'classifier__n_estimators': [30, 50, 75, 90, 100],
                    'classifier__max_depth': [1, 2, 3, 4, 5]
                }
            },
            'DecisionTree': {
                'classifier': DecisionTreeClassifier(),
                'param_grid': {
                    'classifier__max_depth': range(1, 12),
                    'classifier__min_samples_split': range(6, 21, 2),
                    'classifier__min_samples_leaf': range(1, 11),
                    'classifier__max_features': ['sqrt', 'log2', None]
                }
            },
        }
        
        if use_shap_compatible:
            classifiers = {k: v for k, v in classifiers.items() if self.is_supported_by_tree_explainer(k)}

        return classifiers

    def _fix_odd_test_set(self, X_train, y_train, X_test, y_test):
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
        classifier_scores, accuracy_scores, precision_scores, recall_scores = [], [], [], []
        f1_scores, auc_roc_scores, log_loss_scores, confusion_matrices, all_importance_dfs = [], [], [], [], []

        all_X = []
        combined_shap_values = []


        if len(bdf.index) < MINIMUM_SAMPLE_SIZE:
            splitter = LeaveOneOut()
            split = splitter.split(bdf, y)
            round_set_size = False

        else:
            splitter = StratifiedKFold(n_splits=5)
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
                X_train, y_train, X_test, y_test = self._fix_odd_test_set(X_train, y_train, X_test, y_test)
            grid_search = GridSearchCV(pipe, param_grid_classifier, cv=splitter, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            best_classifier = grid_search.best_estimator_.named_steps['classifier']
            if use_shap is True:
                explainer = shap.TreeExplainer(best_classifier)
                shap_values = explainer.shap_values(X_train)
                if isinstance(shap_values, list):  # If there are multiple classes
                    # Here I'm taking only the SHAP values for class 1, which is typical for binary classification.
                    # If you have a multi-class problem, this needs to be adjusted.
                    shap_values = shap_values[1]
                combined_shap_values.append(shap_values)
                all_X.append(X_train)


            if hasattr(classifier, 'feature_importances_'):
                feature_importances = grid_search.best_estimator_.named_steps['classifier'].feature_importances_
                importance_df = pd.DataFrame({
                    'Feature': X_train.columns,
                    'Importance': feature_importances
                })

                # Append this dataframe to the list
                all_importance_dfs.append(importance_df)

                # Your existing code remains here...
            else:
                print(f"{type(classifier).__name__} doesn't have feature_importances_ attribute.")


            train_accuracy = grid_search.score(X_train, y_train)
            print("Training accuracy...", train_accuracy)
            current_best_score = grid_search.best_score_
            current_best_params = grid_search.best_params_
            print("Current best score:", current_best_score)
            print("Current best params:", current_best_params)

            # Compute metrics

            y_pred = grid_search.predict(X_test)
            y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
            
            confusion_matrices.append(confusion_matrix(y_test, y_pred))
            accuracy_scores.append(metrics.accuracy_score(y_test, y_pred))
            precision_scores.append(metrics.precision_score(y_test, y_pred, average='weighted'))
            recall_scores.append(metrics.recall_score(y_test, y_pred, average='weighted'))
            f1_scores.append(metrics.f1_score(y_test, y_pred, average='weighted'))
            if len(np.unique(y_test)) > 1:
                auc_roc_scores.append(metrics.roc_auc_score(y_test, y_pred_proba))
                log_loss_scores.append(metrics.log_loss(y_test, y_pred_proba))

            classifier_scores.append(current_best_score)

        average_score = np.mean(classifier_scores)

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

        if use_shap is True:
            aggregated_shap_values = np.concatenate(combined_shap_values, axis=0)
            combined_X = pd.concat(all_X, axis=0).reset_index(drop=True)

            cpc = PredictorScore.where(classifier_name=classifier_name, predictor_id=self.id, score_type="ShapleySummary")
            if len(cpc) == 0:
                cpc = PredictorScore.find_or_create(classifier_name=classifier_name,score_type="ShapleySummary",predictor_id=self.id)
            else:
                cpc = cpc[0]
            cpc.set_shap_matrix(aggregated_shap_values=aggregated_shap_values, combined_X=combined_X)
        
        average_accuracy_scores = np.mean(accuracy_scores)
        average_precision_scores = np.mean(precision_scores)
        print("Test accuracy...", average_accuracy_scores)
        average_recall_scores = np.mean(recall_scores)
        average_f1_scores = np.mean(f1_scores)
        average_auc_roc_scores = np.mean(auc_roc_scores)
        average_log_loss_scores = np.mean(log_loss_scores)
        median_confusion_matrix = self.median_confusion_matrix(confusion_matrices, accuracy_scores)


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
        extra_metrics["Important Features"] = top_5_median_features
        extra_metrics["Accuracy:"] = average_accuracy_scores
        extra_metrics["Precision:"] = average_precision_scores
        extra_metrics["Recall:"] = average_recall_scores
        extra_metrics["F1-score"] = average_f1_scores
        extra_metrics["AUC-ROC"] = average_auc_roc_scores
        extra_metrics["Log loss"] = average_log_loss_scores
        extra_metrics["Median Confusion Matrix"] = median_confusion_matrix.tolist()

        scores[classifier_name] = average_score
        params[classifier_name] = current_best_params
        acc_metrics[classifier_name] = extra_metrics
        log_loss_scores
        return classifier_scores, grid_search.best_params_, extra_metrics, scores, params, acc_metrics

    # Helper functions for _train_classifier
    # Including:
    # - _split_data: To split data into training and test sets
    # - _fix_odd_test_set: To fix the test set if it has an odd size
    # - _compute_metrics: To compute various metrics
    # - _calculate_average_metrics: To calculate average metrics

    def _update_accuracies(self, classifier_accuracies, classifier_params, classifier_metrics):
        # Update or add classifier accuracies, parameters, and metrics
        self.accuracies = self.multiple_deserialize(self.accuracies)

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




    def fit_model(self, bdf, y, param_grid):
        pipe = Pipeline(steps=[('classifier', RandomForestClassifier())])
        # skf = StratifiedKFold(n_splits=5)

        if len(bdf.index) < MINIMUM_SAMPLE_SIZE:
            counts = LeaveOneOut()
        else:
            counts = StratifiedKFold(n_splits=5)
        groups = bdf['patient']
        scores = []

        for train_index, test_index in counts.split(bdf, y, groups):
            X_train, X_test = bdf.iloc[train_index].drop('is_dominant', axis=1), bdf.iloc[test_index].drop('is_dominant', axis=1)
            y_train, y_test = y[train_index], y[test_index]

            # Convert y_train and y_test to pandas Series
            y_train, y_test = pd.Series(y_train, index=train_index), pd.Series(y_test, index=test_index)

            X_train = X_train.reset_index(drop=True)
            y_train = y_train.reset_index(drop=True)
            X_test = X_test.reset_index(drop=True)
            y_test = y_test.reset_index(drop=True)

            # Check if the test set size is odd
            if len(test_index) % 2 != 0:
                # Count the occurrence of each patient in the test set
                patient_counts = Counter(X_test['patient'])
                # Find the patient that occurs only once
                single_occurrence_patient = [patient for patient, count in patient_counts.items() if count == 1]
                
                if single_occurrence_patient:
                    # Find the index of the single occurrence patient in the test set
                    single_patient_index = X_test[X_test['patient'] == single_occurrence_patient[0]].index[0]

                    # Move this patient from te st set to train set
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
                    print("done fixing!!")

            # Convert y_train and y_test back to numpy arrays, if needed
            y_train, y_test = np.array(y_train), np.array(y_test)

            print("Train set:", X_train)
            print("Test set:", X_test)

            grid_search = GridSearchCV(pipe, param_grid, cv=skf, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            # Get the best score and parameters
            current_best_score = grid_search.best_score_
            current_best_params = grid_search.best_params_
            print("Current best score:", current_best_score)
            print("Current best params:", current_best_params)

            scores.append(current_best_score)

        average_score = np.mean(scores)

        self.accuracies = {'average_score': average_score,'best_params': current_best_params,'features': list(bdf.columns),}
        self.save()


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

    def get_predictor_scores(self, score_type=None):
        # Query the predictor_score table where predictor_id matches this instance's id
        if score_type is None:
            return PredictorScore.where(predictor_id=self.id)
        else:
            return PredictorScore.where(predictor_id=self.id, score_type=score_type)


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
