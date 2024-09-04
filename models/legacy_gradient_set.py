from importlib import import_module
import io
from models.legacy_position_set import PositionSet
import pickle
import dill
from typing import Any, List
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel

from datetime import datetime
from models.legacy_sub_gradient import SubGradient
from models.legacy_position_set import PositionSet
from tsfresh import extract_features
import pandas as pd
import numpy as np
from pandas.compat import pickle_compat
import pdb
from tsfresh.feature_extraction import ComprehensiveFCParameters
from new_pickle import CustomUnpickler, unpickle_file, pickle_data, inspect_pickle  # Import functions from custom_unpickler



from exp_motion_sample_trial import ExpMotionSampleTrial
from motion_filter import MotionFilter

class GradientSet(LegacyBaseModel):
    table_name = "gradient_set"

    def __init__(self, id=None, name=None, sensor_id=None, trial_id=None, matrix=None, aggregated_stats=None, created_at=None, updated_at=None,  set_stats_norm=None, set_stats_non_norm=None, set_stats_abs=None, normalized=None, abs_val=None):
        self.id = id
        self.name = name
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix
        self.aggregated_stats = aggregated_stats
        self.created_at = created_at
        self.updated_at = updated_at
        self.set_stats_norm = set_stats_norm
        self.set_stats_non_norm = set_stats_non_norm
        self.set_stats_abs = set_stats_abs
        self.normalized = normalized
        self.abs_val = abs_val

    # Splits the series based on zero value crossing.
    def get_sub_tasks(self):
        if not self.sensor_id:
            return None
        from importlib import import_module
        Sensor = import_module("models.legacy_sensor").Sensor

        data = self.get_matrix("matrix")
        name = Sensor.get(self.sensor_id).name
        est = ExpMotionSampleTrial(name, name, grad=data)

        return est.valid_sub_tasks()


    def get_sensor_name(self):
        self._cursor.execute("""
            SELECT name FROM sensor
            WHERE id = ?
        """, (self.sensor_id,))
        
        row = self._cursor.fetchone()
        return row[0] if row else None


    def get_valid_tasks(self):
        self.sub_tasks = self.get_sub_tasks()
        MotionFilter.get_valid_motions(self)
    
    def get_sql_row(self):
        self._cursor.execute("""
            SELECT * FROM gradient_set WHERE id = ?
        """, (self.id,))

        row = self._cursor.fetchone()
        if row:
            print("SQL Row for object with id", self.id, ":")
            for col, val in zip([column[0] for column in self._cursor.description], row):
                print(f"{col}: {val}")
        else:
            print(f"No row found for object with id {self.id}")

    def get_task(self):
        Task = import_module("models.legacy_task").Task
        self._cursor.execute("""
            SELECT task.* FROM task
            JOIN patient_task ON task.id = patient_task.task_id
            JOIN trial ON trial.patient_task_id = patient_task.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Task(*row) if row else None

    def get_patient(self):
        Patient = import_module("models.legacy_patient").Patient
        self._cursor.execute("""
            SELECT patient.* FROM patient
            JOIN patient_task ON patient.id = patient_task.patient_id
            JOIN trial ON trial.patient_task_id = patient_task.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Patient(*row) if row else None

    def get_position_set(self):
        return PositionSet.where(trial_id=self.trial_id, sensor_id=self.sensor_id)[0]
        
    # Splits the series based on zero value crossing.
    def split_series(self) -> Any:
        series = self.gradients[self.tasks]
        split_indices = []

        for i in range(1, len(series)):
            if (series.iloc[i] > 0 and series.iloc[i - 1] < 0) or (series.iloc[i] < 0 and series.iloc[i - 1] > 0):
                split_indices.append(i)
        start = 0
        for end in split_indices:
            yield series[start:end]
            start = end

        return series[start:]

    def add_sensor(self, sensor):
        if self.sensor_id == sensor.id:
            print("This GradientSet is already associated with the provided sensor.")
            return

        self.sensor_id = sensor.id
        self.update(sensor_id=self.sensor_id)
        print(f"Sensor with ID {sensor.id} has been associated with this GradientSet.")

    def get_patient_task_id(self):
        self._cursor.execute("SELECT patient_task_id FROM gradient_set WHERE id=?", (self.id,))
        return self._cursor.fetchone()[0]

    def get_patient_task(self):
        PatientTask = import_module("models.legacy_patient_task").PatientTask
        patient_task_id = self.get_patient_task_id()
        return PatientTask.get(patient_task_id)

    def create_subgradients(self):
        print("CREATING SUBS")
        matrix = self.mat()
        subgradients = []
        start_time = 0
        position_set = PositionSet.where(name=self.name,trial_id=self.trial_id,sensor_id=self.sensor_id)
        print(position_set)

        p_matrix = PositionSet.where(name=self.name,trial_id=self.trial_id,sensor_id=self.sensor_id)[0].mat()


        for i in range(1, len(matrix)):
            if (matrix.iloc[i] >= 0 and matrix.iloc[i - 1] < 0) or (matrix.iloc[i] <= 0 and matrix.iloc[i - 1] > 0):
                stop_time = i - 1
                current_slice = matrix.loc[matrix.index[start_time]:matrix.index[stop_time]]
                p_slice = p_matrix.loc[matrix.index[start_time]:matrix.index[stop_time]]
                valid = self.is_valid(current_slice, p_slice)

                if len(SubGradient.where(gradient_set_id=self.id, gradient_set_ord=len(subgradients), name=self.name)) != 0:
                    print("sg already exists...")
                else:
                    subgradient = SubGradient.find_or_create(
                        name=self.name,
                        valid=valid,
                        matrix=current_slice,
                        gradient_set_id=self.id,
                        gradient_set_ord=len(subgradients),
                        start_time=matrix.index[start_time],
                        stop_time=matrix.index[stop_time],
                        mean=current_slice.mean(),
                        median=current_slice.median(),
                        stdev=current_slice.std(),
                        normalized = SubGradient.normalize(current_slice),
                    )   
                    
                    normalized_ts_stats = SubGradient.gen_normalized_tsfresh_stats()
                    non_normalized_ts_stats = SubGradient.gen_tsfresh_stats_non_normalized()
                    abs_val_ts_stats = SubGradient.gen_tsfresh_stats_abs()
                    # pos_ts_stats = SubGradient.get_tsfresh_stats_position(subgradient, manual_position_data=p_slice)
                    subgradient.update(
                        submovement_stats=normalized_ts_stats,
                        submovement_stats_nonnorm=non_normalized_ts_stats,
                        submovement_stats_abs=abs_val_ts_stats,
                    )
                    subgradients.append(subgradient)
                    start_time = i
        # print("created subgrads")
        return subgradients

    def is_valid(self, current_slice, p_slice):
        duration = current_slice.index.stop - current_slice.index.start
        if duration < MotionFilter.DURATION:
            return False
        elif abs(current_slice.mean()) < MotionFilter.VELOCITY:
            return False
        elif abs(float(p_slice.max() - p_slice.min())) < MotionFilter.DISPLACEMENT:
            return False
        return True
    

    @staticmethod
    def unpickle_data(pickled_data):
        if pickled_data is None:
            return None
        try:
            return pickle.loads(pickled_data)
        except ModuleNotFoundError as e:
            return pickle_compat.loads(pickled_data)

    
    def get_matrix(self):
        return pd.Series(self.unpickle_data(self.matrix))

    def get_aggregated_stats(self):
        return pd.DataFrame(self.unpickle_data(self.aggregated_stats))

    def get_set_stats_norm(self):
        return pd.DataFrame(self.unpickle_data(self.set_stats_norm))
        
    def get_set_stats_non_norm(self):
        return pd.DataFrame(self.unpickle_data(self.set_stats_non_norm))

    def get_set_stats_abs(self):
        return pd.DataFrame(self.unpickle_data(self.set_stats_abs))

    def mat(self):
        return pd.Series(pickle.loads(self.matrix))
    
    def gen_set_stats(self, force=False):
        stats_to_update = {}

        if self.set_stats_non_norm is None or force:
            stats_to_update['set_stats_non_norm'] = self.gen_non_norm_set_stats(force=force, collect_only=True)
        
        if self.set_stats_norm is None or force:
            stats_to_update['set_stats_norm'] = self.gen_norm_set_stats(force=force, collect_only=True)
        
        if self.set_stats_abs is None or force:
            stats_to_update['set_stats_abs'] = self.gen_abs_set_stats(force=force, collect_only=True)
        
        if stats_to_update:
            self.update(**stats_to_update)
        
        print("DONE")

    def fix_reg_sub_stats(self, force=False):
        if self.aggregated_stats is None or force:
            print("fixing non norm")
            self.set_aggregate_non_norm_stats()
        
        if self.normalized is None or force:
            print("fixing norm")
            self.set_aggregate_normalized_stats()
        
        if self.abs_val is None or force:
            print("fixing abs")
            self.set_aggregate_abs_val_stats()

        print("DONE")
    
    def get_non_norm_set_stats(self):
        obj = pickle.loads(self.set_stats_non_norm)

        if obj is not None:
            print("Pickle data loaded successfully with custom dill unpickler.")
            return obj
        else:
            print("Failed to load pickle data with both pandas and custom dill unpickler.")
            return None
        
    
    def get_abs_set_stats(self):
        return pickle.loads(self.set_stats_abs)
        if obj is not None:
            print("Pickle data loaded successfully with custom dill unpickler.")
            return obj
        else:
            print("Failed to load pickle data with both pandas and custom dill unpickler.")
            return None

    def get_norm_set_stats(self, use_sql=False):
        if use_sql:
            return pickle.loads(self._cursor.execute("""SELECT set_stats_norm FROM gradient_set WHERE id = ?""", (self.id,)).fetchone()[0])
        
        return unpickle_file(self.set_stats_norm)

    def gen_non_norm_set_stats(self, force=False, collect_only=False):
        if force or self.set_stats_non_norm is None:
            movement = pd.DataFrame(self.mat())
            movement["id"] = self.id
            movement["samplepoint"] = range(len(movement))
            set_stats_non_norm = self.extract_features_from_movement(movement)
            if collect_only:
                return set_stats_non_norm
            self.update(set_stats_non_norm=set_stats_non_norm)
            print("done non_norm")
            print("Updated set_stats_non_norm:", set_stats_non_norm)
        else:
            print("set_stats_non_norm already set")
    
    def gen_abs_set_stats(self, force=False, collect_only=False):
        if force or self.set_stats_abs is None:
            movement = abs(pd.DataFrame(self.mat()))
            movement["id"] = self.id
            movement["samplepoint"] = range(len(movement))
            set_stats_abs = self.extract_features_from_movement(movement)
            if collect_only:
                return set_stats_abs
            self.update(set_stats_abs=set_stats_abs)
            print("done abs")
            print("Updated set_stats_abs:", set_stats_abs)
        else:
            print("set_stats_abs already set")

    def gen_norm_set_stats(self, force=False, collect_only=False):
        if force or self.set_stats_norm is None:
            if self.normalized is None:
                movement = pickle.loads((self.normalize_to_length_3000()))
            else:
                movement = pd.DataFrame(self.normalized)
            movement["id"] = self.id
            movement["samplepoint"] = range(len(movement))
            set_stats_norm = self.extract_features_from_movement(movement)
            if collect_only:
                return set_stats_norm
            self.update(set_stats_norm=set_stats_norm)
            print("done norm")
            print("Updated set_stats_norm:", set_stats_norm)
        else:
            print("set_stats_norm already set")
    
    def normalize_to_length_3000(self):
        data = self.mat()
        normed_amplitude = abs(data / np.max(np.abs(data)))
        start, end = normed_amplitude.index[0], normed_amplitude.index[-1]

        if end != start:
            x_vals = np.linspace(start, end, 3000)
        else:
            x_vals = [start] * 3000
        normed_temporally = np.interp(x_vals, normed_amplitude.index.tolist(), normed_amplitude)
        normed_temporally = pd.DataFrame({"grad_data": normed_temporally}, index=x_vals)
        normed_temporally["samplepoint"] = x_vals

        return pickle.dumps(normed_temporally)

    @staticmethod
    def extract_features_from_movement(movement, params=ComprehensiveFCParameters()):
        extracted_features = extract_features(
            movement, 
            column_id='id', 
            column_sort='samplepoint', 
            n_jobs=1, 
            default_fc_parameters=params
        )
        print("Extracted features:", extracted_features)
        return memoryview(pickle.dumps(extracted_features))

    def get_ts_fresh_stats(self, force=False):
        print("yo")


    def mat_df(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        series = pd.Series(pickle.loads(self.matrix))
        
        # Convert the pandas Series to a DataFrame
        dataframe = series.to_frame(name='value')
        
        # Reset the index and add a new column for the time points
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'index': 'time'}, inplace=True)

        return dataframe

    def deserialize_matrix(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pickle
    
    def sub_gradients(self):
        from importlib import import_module
        SubGradient = import_module("models.legacy_sub_gradient").SubGradient

        return SubGradient.where(gradient_set_id=self.id)

    @classmethod
    def create_all_available_sub_gradients(cls):
        all_gs = GradientSet.all()
        created_sg = []
        i = 0
        for gs in all_gs:
            # Skip this gradient set if it already has subgradients
            if len(sg.subgsub_gradients()) != 0:
                print(f"sub_gradients already exist for gs {gs.id}")
                continue
            else:
                sg = gs.create_subgradients()
                i += 1
                created_sg += sg
        print("done")

        return created_sg

    def calc_aggregate_stats(self, abs_val=False, non_normed=False, mv=True, save_attr=False, set_only=False):
        sub_stats_all = pd.DataFrame()
        subgrads = self.sub_gradients()
        normalized = not non_normed
        for subgrad in subgrads:
            if subgrad.valid:
                sub_stats = subgrad.get_sub_stats(normalized=normalized, abs_val=abs_val)
                sub_stats_all = pd.concat([sub_stats_all, sub_stats])

        # for each stat type in the submovement stats, calculate aggregate stat for the whole trial
        stats = pd.DataFrame()
        for colname, colvalues in sub_stats_all.items():
            aggregate = {"mean": np.mean(colvalues)}
            stats = pd.concat([stats, pd.DataFrame([aggregate], index=[colname])])

        if save_attr:
            if normalized and not abs_val:
                print("updating norm...")
                self.update(normalized=pickle.dumps(stats))
            elif abs_val is False and not normalized:
                print("updating non_norm...")
                self.update(aggregated_stats=pickle.dumps(stats))
            elif abs_val is True:
                print("updating abs_val...")
                self.update(abs_val=pickle.dumps(stats))
            else:
                print("need to update")
            if set_only:
                return

        if mv:
            return memoryview(pickle.dumps(stats))
        else:
            return stats

    def set_aggregate_normalized_stats(self):
        self.calc_aggregate_stats(non_normed=False, abs_val=False, mv=False, save_attr=True, set_only=True)
        print("set norm for gs", self.id)

    def set_aggregate_abs_val_stats(self):
        self.calc_aggregate_stats(non_normed=True, abs_val=True, mv=False, save_attr=True, set_only=True)
        print("set abs val for gs", self.id)

    def set_aggregate_non_norm_stats(self):
        self.calc_aggregate_stats(non_normed=True, abs_val=False, save_attr=True, set_only=True)
        print("set non norm for gs", self.id)

    def get_aggregate_normalized_stats(self):
        if self.normalized is None:
            return self.calc_aggregate_stats(non_normed=False, abs_val=False, mv=False, save_attr=True)
        elif pickle.loads(self.normalized).empty:
            return self.calc_aggregate_stats(non_normed=False, abs_val=False, mv=False, save_attr=True)
    
        return pickle.loads(self.normalized)

    def get_aggregate_abs_val_stats(self):
        if self.abs_val is None:
            return self.calc_aggregate_stats(non_normed=False, abs_val=True, mv=False, save_attr=True)
        elif pickle.loads(self.normalized).empty:
            return self.calc_aggregate_stats(non_normed=False, abs_val=True, mv=False, save_attr=True)
    
        return pickle.loads(self.abs_val)

    def get_aggregate_non_norm_stats(self, abs_val=False, non_normed=True, mv=True):
        ags = self.calc_aggregate_stats(abs_val=abs_val, non_normed=non_normed, mv=mv)

        return ags

    def view_3d(self):
        from importlib import import_module
        ShapeRotator = import_module("viewers.shape_rotator").ShapeRotator
        ShapeRotator.plot_3d_sg(self)


        
    # def get_tsfresh_data(self):
    #     matrix_df = self.mat_df()
    #     matrix_df['id'] = 0
    #     features = extract_features(matrix_df, column_id='id', column_sort='time')
    #     # print(features.describe())
