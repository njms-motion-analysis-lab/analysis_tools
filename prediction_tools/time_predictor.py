# models/time_predictor.py

import json
import pickle

import numpy as np
import pandas as pd
import shap
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient import Patient
from prediction_tools.predictor_score import PredictorScore
from prediction_tools.legacy_predictor import Predictor
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from transitions import Machine
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

MINIMUM_SAMPLE_SIZE = 10
DEFAULT_K_FOLD_SPLITS = 5

REGRESSION_MODELS = {
    "RandomForestRegressor": {
        "regressor": RandomForestRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__max_depth": [None, 10],
            "regressor__min_samples_split": [2],
            "regressor__min_samples_leaf": [1, 2],
            "regressor__max_features": ["sqrt", "log2"]
        }
    },
    "XGBRegressor": {
        "regressor": XGBRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__max_depth": [3, 6],
            "regressor__learning_rate": [0.01, 0.1],
            "regressor__subsample": [0.8, 1.0]
        }
    },
    "CatBoostRegressor": {
        "regressor": CatBoostRegressor(random_state=42, verbose=0),
        "param_grid": {
            "regressor__iterations": [100, 200],
            "regressor__depth": [4, 6],
            "regressor__learning_rate": [0.01, 0.1]
        }
    },
    # ---- NEW MODELS BELOW ----
    "LGBMRegressor": {
        "regressor": LGBMRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__max_depth": [3, 6, 10],
            "regressor__learning_rate": [0.1, 0.3],
            "regressor__num_leaves": [10, 31] 
        }
    },
    "ExtraTreesRegressor": {
        "regressor": ExtraTreesRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__max_depth": [None, 10],
            "regressor__min_samples_split": [2],
            "regressor__min_samples_leaf": [1, 2],
            "regressor__max_features": ["sqrt", "log2"]
        }
    },
    "AdaBoostRegressor": {
        "regressor": AdaBoostRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__learning_rate": [0.01, 0.1, 1.0],
            # If needed, you can also set 'base_estimator' to a small DecisionTreeRegressor
        }
    },
    "GradientBoostingRegressor": {
        "regressor": GradientBoostingRegressor(random_state=42),
        "param_grid": {
            "regressor__n_estimators": [50, 100],
            "regressor__learning_rate": [0.01, 0.1],
            "regressor__max_depth": [3, 6],
            "regressor__subsample": [0.8, 1.0],
            "regressor__max_features": ["sqrt", "log2"]
        }
    },
    "SVR": {
        "regressor": SVR(),
        "param_grid": {
            "regressor__kernel": ["rbf", "linear"],
            "regressor__C": [0.1, 1, 10],
            "regressor__gamma": ["scale", 0.01, 0.1]
        }
    },
    "MLPRegressor": {
        "regressor": MLPRegressor(random_state=42, max_iter=1000),
        "param_grid": {
            "regressor__hidden_layer_sizes": [(50,), (100,)],
            "regressor__activation": ["relu", "tanh"],
            "regressor__alpha": [1e-4, 1e-2],  # L2 penalty
            "regressor__learning_rate_init": [0.001, 0.01]
        }
    },
}



class TimePredictor(LegacyBaseModel):
    """
    A class for predicting continuous values (e.g. time to complete a task).
    Mirrors much of the design of the existing `Predictor` class, but for regression.
    """
    table_name = "time_predictor"

    def __init__(
        self, 
        id=None,
        task_id=None,
        sensor_id=None,
        metrics=None,
        matrix=None,
        created_at=None,
        updated_at=None,
        cohort_id=None,
        multi_time_predictor_id=None,
        aggregated_stats=None,
        aggregated_stats_non_normed=None,
        predictor_id=None,
        predictor_feature_id=None,
    ):
        self.id = id
        self.task_id = task_id
        self.sensor_id = sensor_id
        self.metrics = metrics  # JSON or text field to store e.g. MSE, MAE, R², etc.
        self.matrix = matrix    # Could store SHAP values or other data
        self.created_at = created_at
        self.updated_at = updated_at
        self.cohort_id = cohort_id
        self.multi_time_predictor_id = multi_time_predictor_id
        self.aggregated_stats = aggregated_stats
        self.aggregated_stats_non_normed = aggregated_stats_non_normed
        self.predictor_id = predictor_id
        self.predictor_feature_id = predictor_feature_id

    def get_df(self, force_load=False):
        """
        Similar to the `Predictor.get_df()` method.
        Should load or generate the DataFrame containing your features and
        the continuous target (e.g. 'time_needed').
        """
        # Example approach if you stored your aggregated stats in a pickle-serialized field
        if self.aggregated_stats and not force_load:
            return pickle.loads(self.aggregated_stats)

        # Otherwise, generate the DataFrame from scratch:
        df = self._generate_time_data()
        if df is None:
            return None

        # Store it in `aggregated_stats`
        self.aggregated_stats = memoryview(pickle.dumps(df))
        self.update(aggregated_stats=self.aggregated_stats)
        return df

    def non_norm(self):
        return Predictor.get(self.predictor_feature_id).non_norm
    
    def abs_val(self):
        return Predictor.get(self.predictor_feature_id).abs_val
    
    def feature_predictor(self):
        return Predictor.get(self.predictor_feature_id)
    
    def get_continuous_default_predictor_scores(self):
        return self.get_predictor_scores(score_type='continuous_default')
    
    def get_dom_predictor_scores(self):
        return self.get_predictor_scores(score_type='dom')
    
    def get_nondom_predictor_scores(self):
        return self.get_predictor_scores(score_type='nondom')

    gcdps = get_continuous_default_predictor_scores

    def get_predictor_scores(self, score_type=None, model_name=None):
        if score_type is None and model_name is None:
            return PredictorScore.where(time_predictor_id=self.id)
        elif model_name is None:
            return PredictorScore.where(time_predictor_id=self.id, score_type=score_type)
        elif score_type is None:
            return PredictorScore.where(time_predictor_id=self.id, classifier_name=model_name)
        else:
            return []
    
    gps = get_predictor_scores

    def timing_predictor(self):
        return Predictor.get(self.predictor_id)
    
    def multi_time_predictor(self):
        from importlib import import_module
        MultiTimePredictor = import_module("prediction_tools.multi_time_predictor").MultiTimePredictor
        
        return MultiTimePredictor.get(self.multi_time_predictor_id)
    
    def multi_time_predictor_timing_multi_predictor(self):
        from importlib import import_module
        MultiPredictor = import_module("prediction_tools.legacy_multi_predictor").MultiPredictor
        return MultiPredictor.get(self.multi_time_predictor().multi_predictor_id)
    
    def gather_non_norm_timing_length_predictor(self):
        all_timing_preds = self.multi_time_predictor_timing_multi_predictor().get_all_preds()
        timing_mp_predictors = [
            tpred for tpred in all_timing_preds
            if getattr(tpred, 'non_norm', 0) == 1
        ]
    
        matching_times = [
            tpred for tpred in timing_mp_predictors
            if (tpred.task_id == self.task_id and
                tpred.sensor_id == self.sensor_id and
                tpred.cohort_id == self.cohort_id)
        ]

        return matching_times[0]


    def _generate_time_data(self, time_needed_as_log=False, special_case=None):
        """
        Custom logic to build a DataFrame that combines:
        - final_feature_df (trimmed sub-movement features)
        - final_time_df    (trimmed fill-movement features)
        - timing_df        (full timing data, used to get 'grad_data__length_y')
        and produces a final DataFrame with 'time_needed' as the continuous target.

        If time_needed_as_log=True, it also creates 'time_needed_log'
        as a log1p transform of the original 'time_needed'.

        If special_case='dom', only keep rows where is_dominant == 1.
        If special_case='nondom', only keep rows where is_dominant == 0.
        Otherwise do nothing special with is_dominant.

        If final_feature_df == final_time_df, we create a copy of final_time_df
        so that rename/drop steps don't mutate final_feature_df in-place.
        If final_feature_df and final_time_df differ in lengths, we find
        the (patient, is_dominant) intersection in all 3 DataFrames and keep only those.
        """

        def get_patient_id(p_name):
            # Convert a patient name/string to an ID from the DB, if needed
            patient = Patient.where(name=p_name)
            if len(patient) != 0:
                return patient[0].id
            else:
                print("MISSING PATIENT ID")
                return None

        # 1. Load relevant DataFrames
        final_feature_df = self.feature_predictor().get_final_bdf()[0]
        final_time_df    = self.timing_predictor().get_final_bdf()[0]
        timing_df        = self.gather_non_norm_timing_length_predictor().get_bdf()

        # Convert timing_df['patient'] if needed
        timing_df['patient'] = timing_df['patient'].astype(str).apply(get_patient_id)

        if final_feature_df is None or final_time_df is None or timing_df is None:
            print("One of the required DataFrames (feature, final_time, or timing) is None.")
            return pd.DataFrame()

        # 2. Ensure each trimmed DF has 'patient' & 'is_dominant'
        for df_name, df in [('feature', final_feature_df), ('time', final_time_df)]:
            if not all(col in df.columns for col in ['patient', 'is_dominant']):
                print(f"{df_name} DataFrame missing 'patient' or 'is_dominant'. Returning empty.")
                return pd.DataFrame()

        # 2b. If special_case='dom' or 'nondom', filter final_feature_df, final_time_df, and timing_df
        if special_case == 'dom':
            final_feature_df = final_feature_df[final_feature_df['is_dominant'] == 1]
            final_time_df    = final_time_df[final_time_df['is_dominant'] == 1]
            timing_df        = timing_df[timing_df['is_dominant'] == 1]
        elif special_case == 'nondom':
            final_feature_df = final_feature_df[final_feature_df['is_dominant'] == 0]
            final_time_df    = final_time_df[final_time_df['is_dominant'] == 0]
            timing_df        = timing_df[timing_df['is_dominant'] == 0]

        # 3. If final_feature_df and final_time_df are actually the same object, create a copy of time
        if final_feature_df is final_time_df:
            print("final_feature_df and final_time_df are the same object. Creating a copy of final_time_df.")
            final_time_df = final_time_df.copy()

        # 4. Check if lengths differ; if so, do intersection logic on (patient, is_dominant)
        from itertools import product
        len_feat = len(final_feature_df)
        len_time = len(final_time_df)

        if len_feat != len_time:
            print(f"feature_df (len={len_feat}) and time_df (len={len_time}) differ in length. Doing intersection on (patient,is_dominant).")

            feat_set = set(zip(final_feature_df['patient'], final_feature_df['is_dominant']))
            time_set = set(zip(final_time_df['patient'], final_time_df['is_dominant']))
            timing_set = set(zip(timing_df['patient'], timing_df['is_dominant']))

            # Intersection only combos that appear in all three
            common_rows = feat_set & time_set & timing_set
            print(f"Keeping {len(common_rows)} combos in all three dataframes.")

            # Filter each DataFrame
            final_feature_df = final_feature_df[
                final_feature_df[['patient','is_dominant']].apply(tuple, axis=1).isin(common_rows)
            ]
            final_time_df = final_time_df[
                final_time_df[['patient','is_dominant']].apply(tuple, axis=1).isin(common_rows)
            ]
            timing_df = timing_df[
                timing_df[['patient','is_dominant']].apply(tuple, axis=1).isin(common_rows)
            ]

        # 5. Drop 'patient'/'is_dominant' from final_time_df (already in final_feature_df)
        final_time_df = final_time_df.drop(columns=['patient', 'is_dominant'], errors='ignore')

        # 6. Remove unneeded length columns from final_time_df
        cols_to_remove = ['grad_data__length_x', 'grad_data__length_y', 'grad_data__length_z']
        final_time_df = final_time_df.drop(columns=cols_to_remove, errors='ignore')

        # 7. Rename columns that start with 'grad_data__' -> 'grad_data__gs_'
        rename_map = {}
        for col in final_time_df.columns:
            if col.startswith('grad_data__'):
                rename_map[col] = col.replace('grad_data__', 'grad_data__gs_')
        final_time_df = final_time_df.rename(columns=rename_map)

        # 8. Concatenate horizontally
        combined_df = pd.concat(
            [final_feature_df.reset_index(drop=True),
            final_time_df.reset_index(drop=True)],
            axis=1
        )

        # 9. Ensure we have 'grad_data__length_y' in timing_df
        if 'grad_data__length_y' not in timing_df.columns:
            print("timing_df lacks 'grad_data__length_y'. Returning empty.")
            return pd.DataFrame()

        # Possibly remove missing patient IDs in timing_df
        timing_df = timing_df.dropna(subset=['patient'])

        # 9b. If timing_df length differs from final_feature_df, do intersection again
        if len(timing_df) != len(final_feature_df):
            feat_set2 = set(zip(final_feature_df['patient'], final_feature_df['is_dominant']))
            time_set2 = set(zip(timing_df['patient'], timing_df['is_dominant']))
            common2 = feat_set2 & time_set2

            final_feature_df = final_feature_df[
                final_feature_df[['patient','is_dominant']].apply(tuple, axis=1).isin(common2)
            ]
            final_time_df = final_time_df.iloc[:len(final_feature_df)]  # or do a more robust match
            timing_df = timing_df[
                timing_df[['patient','is_dominant']].apply(tuple, axis=1).isin(common2)
            ]

        # 9c. Add 'time_needed' from timing_df. We assume row alignment after the intersections.
        combined_df['time_needed'] = timing_df['grad_data__length_y'].reset_index(drop=True)

        # 10. Convert 'patient' to int if present
        if 'patient' in combined_df.columns:
            combined_df['patient'] = combined_df['patient'].astype(int, errors='ignore')

        # 11. (Optional) Log transform
        if time_needed_as_log:
            combined_df['time_needed_log'] = np.log1p(combined_df['time_needed'])

        # 12. Reorder columns so the last columns are [time_needed, (optionally time_needed_log), patient]
        reorder_cols = list(combined_df.columns)
        for special_col in ['time_needed', 'time_needed_log', 'patient']:
            if special_col in reorder_cols:
                reorder_cols.remove(special_col)

        if time_needed_as_log and 'time_needed_log' in combined_df.columns:
            reorder_cols += ['time_needed', 'time_needed_log', 'patient']
        else:
            reorder_cols += ['time_needed', 'patient']

        combined_df = combined_df[reorder_cols]

        # Remove rows missing 'time_needed'
        combined_df = combined_df.dropna(subset=['time_needed'])

        print(f"Final combined_df shape: {combined_df.shape}")
        return combined_df


    def train_regression(self, use_log=False, use_shap=True, special_case='continuous_default'):
        """
        Train multiple regression models (e.g. RandomForestRegressor, XGBRegressor, CatBoostRegressor)
        on the data and store results (MSE, MAE, R²) in self.metrics or in predictor_score.

        If use_log=True, we train on 'time_needed_log' and drop 'time_needed' from X.
        If use_log=False, we train on 'time_needed' and drop 'time_needed_log' if present.

        special_case can be:
        - 'dom': use only is_dominant == 1 rows (regular KFold, row-based split)
        - 'nondom': use only is_dominant == 0 rows (regular KFold, row-based split)
        - 'continuous_default': no filtering by is_dominant, 
            but we do GroupKFold by patient and a patient-based final holdout split.
        """
        df = self.get_df(force_load=True)
        if df is None or df.empty:
            print("No data found for TimePredictor")
            return None

        # 1. Filter if special_case is 'dom' or 'nondom'
        if special_case == 'dom':
            df = df[df['is_dominant'] == 1]
        elif special_case == 'nondom':
            df = df[df['is_dominant'] == 0]

        if df.empty:
            print(f"No rows left after filtering with special_case='{special_case}'.")
            return None

        # 2. Decide which target column to use
        if use_log:
            if 'time_needed_log' not in df.columns:
                print("use_log=True but 'time_needed_log' is missing.")
                return None
            y = df['time_needed_log']
            drop_cols = ['time_needed'] if 'time_needed' in df.columns else []
        else:
            if 'time_needed' not in df.columns:
                print("Missing 'time_needed' column for training.")
                return None
            y = df['time_needed']
            drop_cols = ['time_needed_log'] if 'time_needed_log' in df.columns else []

        # 3. Prepare features X
        #    Example: drop patient + is_dominant from features, but keep them if you want them as input
        X = df.drop(columns=(drop_cols + ['patient', 'is_dominant']), errors='ignore')
        # Also remove the column we *are* using as target
        if use_log:
            X = X.drop(columns=['time_needed_log'], errors='ignore')
        else:
            X = X.drop(columns=['time_needed'], errors='ignore')

        if X.empty:
            print("No features left in X after dropping columns. Check your pipeline or DataFrame structure.")
            return None

        # 4. Build a results dict to store each model's performance
        results = {}
        from sklearn.model_selection import (
            KFold, GroupKFold, GridSearchCV, train_test_split
        )
        from sklearn.pipeline import Pipeline
        import numpy as np
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        # Choose cross-validation strategy
        # - If 'continuous_default', we do GroupKFold by patient.
        # - Otherwise, standard KFold.
        if special_case == 'continuous_default':
            if 'patient' not in df.columns:
                print("Cannot do GroupKFold if 'patient' column is missing.")
                return None
            cv = GroupKFold(n_splits=5)
            cv_kwargs = dict(
                X=X,
                y=y,
                groups=df['patient'],  # each group is a patient
            )
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_kwargs = dict(
                X=X,
                y=y
            )

        # from models.regression_config import REGRESSION_MODELS
        for model_name, model_info in REGRESSION_MODELS.items():
            pipe = Pipeline([("regressor", model_info["regressor"])])
            param_grid = model_info["param_grid"]

            # 5. GridSearch with negative MSE as scoring
            grid_search = GridSearchCV(
                pipe,
                param_grid,
                scoring="neg_mean_squared_error",
                cv=cv.split(**cv_kwargs)  # pass splitted folds
            )
            
            grid_search.fit(X, y)
            best_estimator = grid_search.best_estimator_["regressor"]
            best_params = grid_search.best_params_
            best_cv_mse = -grid_search.best_score_  # positive MSE

            # 6. Evaluate on a holdout
            #    If special_case='continuous_default', do a patient-based split
            if special_case == 'continuous_default':
                unique_patients = df['patient'].unique()
                # Keep the same random_state so it's reproducible
                pat_train, pat_test = train_test_split(
                    unique_patients, test_size=0.2, random_state=42
                )
                train_mask = df['patient'].isin(pat_train)
                test_mask  = df['patient'].isin(pat_test)
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

            # Fit best estimator on the new train set
            best_estimator.fit(X_train, y_train)
            y_pred = best_estimator.predict(X_test)

            # If training in log space, revert predictions
            if use_log:
                y_pred_linear = np.expm1(y_pred)
                y_test_linear = np.expm1(y_test)
            else:
                y_pred_linear = y_pred
                y_test_linear = y_test

            mse = mean_squared_error(y_test_linear, y_pred_linear)
            mae = mean_absolute_error(y_test_linear, y_pred_linear)
            r2  = r2_score(y_test_linear, y_pred_linear)

            # Optionally do SHAP
            shap_values = None
            if use_shap:
                try:
                    import shap
                    explainer = shap.TreeExplainer(best_estimator)
                    shap_values = explainer.shap_values(X_test)
                except Exception as e:
                    print(f"SHAP not supported by {model_name}: {e}")

            # Store results in a nested structure
            if model_name not in results:
                results[model_name] = {}
            results[model_name][special_case] = {
                "best_params": best_params,
                "best_cv_mse": best_cv_mse,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "shap_values": shap_values,
            }

            # 7. Save metrics in predictor_score
            self._save_metric_in_predictor_score(model_name, results, special_case, X_test)

        # 8. Save all results in self.metrics as JSON
        import json
        from datetime import datetime

        def default_numpy_handler(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            raise TypeError(f"Object {obj} is not JSON serializable")

        results_json = json.dumps(results, default=default_numpy_handler)

        self.update(
            metrics = results_json,
            updated_at = str(datetime.now())
        )
        

        return results


    def _save_metric_in_predictor_score(self, model_name, results, special_case, X_test):
        from prediction_tools.predictor_score import PredictorScore
        import pickle
        
        # 1. The numeric results go in continuous_results or something similar
        numeric_data = {
            "best_cv_mse": results[model_name][special_case]["best_cv_mse"],
            "mse": results[model_name][special_case]["mse"],
            "mae": results[model_name][special_case]["mae"],
            "r2": results[model_name][special_case]["r2"],
            "best_params": results[model_name][special_case]["best_params"]
            # ... anything else numeric ...
        }
        # self.id = id
        # self.task_id = task_id
        # self.sensor_id = sensor_id
        # self.metrics = metrics  # JSON or text field to store e.g. MSE, MAE, R², etc.
        # self.matrix = matrix    # Could store SHAP values or other data
        # self.created_at = created_at
        # self.updated_at = updated_at
        # self.cohort_id = cohort_id
        # self.multi_time_predictor_id = multi_time_predictor_id
        # self.aggregated_stats = aggregated_stats
        # self.aggregated_stats_non_normed = aggregated_stats_non_normed
        # self.predictor_id = predictor_id
        # self.predictor_feature_id = predictor_feature_id

        # 2. The SHAP array is in shap_values
        shap_values = results[model_name][special_case]["shap_values"]

        # 3. Create or update a predictor_score row for (model_name, special_case)
        existing = PredictorScore.where(
            time_predictor_id=self.id,
            classifier_name=model_name,
            score_type=special_case
        )

        if len(existing) == 0:
            ps_obj = PredictorScore.find_or_create(
                classifier_name=model_name,
                score_type=special_case,
                time_predictor_id=self.id,
            )
        else:
            ps_obj = existing[0]

        # 4. Update 'continuous_results' with numeric_data (JSON or pickled)
        ps_obj.update(
            continuous_results = json.dumps(numeric_data),
            multi_predictor_id = self.multi_time_predictor_id,
            time_predictor_id = self.id,
            multi_time_predictor_id = self.multi_time_predictor_id
        )

        # 5. If you also want to store shap in ps_obj.matrix:
        if shap_values is not None:
            # shap_values is 2D array
            # Suppose you also have X_test stored somewhere or aggregated
            # Let's say you do something like:
            combined_matrix = (shap_values, X_test.values, X_test.columns) 
            c_matrix = pickle.dumps(combined_matrix)
            
            ps_obj.update(matrix = memoryview(c_matrix))

            pass

        # 6. Save
        # ps_obj.save()

    def save(self):
        """
        Saves the current TimePredictor state to DB. 
        Similar to how `Predictor.save()` works.
        """
        updated_rows = self.update(
            id=self.id,
            task_id=self.task_id,
            sensor_id=self.sensor_id,
            metrics=self.metrics,
            matrix=self.matrix,
            created_at=self.created_at,
            updated_at=self.updated_at,
            cohort_id=self.cohort_id,
            multi_time_predictor_id=self.multi_time_predictor_id,
            aggregated_stats=self.aggregated_stats,
            aggregated_stats_non_normed=self.aggregated_stats_non_normed
        )
        self.__class__._conn.commit()
        return updated_rows > 0


class TimePredictorFSM:
    """
    Handles states like untrained -> training -> trained -> evaluating -> complete
    for a TimePredictor pipeline.
    """
    states = ["untrained", "training", "trained", "evaluating", "complete"]
    
    def __init__(self, time_predictor):
        """
        time_predictor is an instance of your TimePredictor class.
        The FSM can call time_predictor methods during transitions.
        """
        self.time_predictor = time_predictor
        self.machine = Machine(
            model=self, 
            states=TimePredictorFSM.states, 
            initial="untrained"
        )
        
        # Add transitions
        self.machine.add_transition(
            trigger="start_training", 
            source="untrained", 
            dest="training", 
            after="on_start_training"
        )
        self.machine.add_transition(
            trigger="finish_training", 
            source="training", 
            dest="trained", 
            after="on_finish_training"
        )
        self.machine.add_transition(
            trigger="start_evaluation", 
            source="trained", 
            dest="evaluating", 
            after="on_start_evaluation"
        )
        self.machine.add_transition(
            trigger="finish_evaluation", 
            source="evaluating", 
            dest="complete", 
            after="on_finish_evaluation"
        )

    # Callbacks (executed after each transition)
    def on_start_training(self):
        """
        Typically you'd call the time_predictor.train_regression() method here
        or set up the environment. 
        """
        print("FSM: Starting training...")
        # Example: run the training pipeline right away:
        results = self.time_predictor.train_regression()
        if results is None:
            print("FSM: Training failed or no data found. Possibly handle error state.")
            # Optional: self.machine.set_state('error')
        else:
            print("FSM: Training in progress or completed. Next step is to call finish_training().")

    def on_finish_training(self):
        """
        Mark training done, e.g. confirm self.time_predictor.metrics was updated.
        """
        print("FSM: Finished training!")
        # self.time_predictor might now contain metrics in self.time_predictor.metrics

    def on_start_evaluation(self):
        """
        Here you might do something like:
        - Evaluate on a holdout set 
        - Or run SHAP analysis 
        - etc.
        """
        print("FSM: Starting evaluation...")
        # Example pseudo-code:
        # shap_values = self.time_predictor.run_additional_shap_evaluation()
        # store results, etc.

    def on_finish_evaluation(self):
        """
        Now we've completed all evaluation steps (if any).
        """
        print("FSM: Finished evaluation and pipeline is complete!")