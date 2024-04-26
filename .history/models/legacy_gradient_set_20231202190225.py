from models.legacy_position_set import PositionSet
import pickle
from typing import Any, List
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_task import Task
from models.legacy_patient import Patient
from datetime import datetime
from models.legacy_sub_gradient import SubGradient
from models.legacy_position_set import PositionSet
from tsfresh import extract_features
import pandas as pd
import numpy as np
import pdb


from models.legacy_patient_task import PatientTask
from exp_motion_sample_trial import ExpMotionSampleTrial
from motion_filter import MotionFilter

class GradientSet(LegacyBaseModel):
    table_name = "gradient_set"

    def __init__(self, id=None, name=None, sensor_id=None, trial_id=None, matrix=None, aggregated_stats=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix
        self.aggregated_stats = aggregated_stats

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

    def get_task(self):
        self._cursor.execute("""
            SELECT task.* FROM task
            JOIN patient_task ON task.id = patient_task.task_id
            JOIN trial ON trial.patient_task_id = patient_task.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Task(*row) if row else None

    def get_patient(self):
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

                    ts_stats = SubGradient.get_tsfresh_stats(subgradient)
                    non_normalized_ts_stats = SubGradient.get_tsfresh_stats_non_normalized(subgradient)
                    pos_ts_stats = SubGradient.get_tsfresh_stats_position(subgradient, manual_position_data=p_slice)
                    subgradient.update(
                        submovement_stats=ts_stats,
                        submovement_stats_nonnorm=non_normalized_ts_stats,
                        submovement_stats_position=pos_ts_stats
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
    

    def mat(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pd.Series(pickle.loads(self.matrix))

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

    def calc_aggregate_stats(self, abs_val=False, non_normed=False):
        sub_stats_all = pd.DataFrame()
        subgrads = self.sub_gradients()
        for subgrad in subgrads:
            if subgrad.valid:
                import pdb;pdb.set_trace()
                normalized = not non_normed
                sub_stats = subgrad.get_sub_stats(normalized=normalized, abs_val=abs_val)
                sub_stats_all = pd.concat([sub_stats_all, sub_stats])
                #sub_stats_all = pd.concat([sub_stats_all, pd.DataFrame([sub_stats])], ignore_index=True)

        # for each stat type in the submovement stats, calculate aggregate stat for the whole trial
        stats = pd.DataFrame()
        for colname, colvalues in sub_stats_all.items():
            # print(colname)
            aggregate = {"mean": np.mean(colvalues), "median": np.median(colvalues), 
                            "sd":np.std(colvalues), "IQR": np.subtract(*np.percentile(colvalues, [75, 25])),
                            "10th": np.percentile(colvalues, 10), "90th": np.percentile(colvalues, 90)}
            #print(aggregate_stats)
            stats = pd.concat([stats, pd.DataFrame([aggregate], index=[colname])])
        return memoryview(pickle.dumps(stats))

    def get_aggregate_stats(self):
        return pickle.loads(self.aggregated_stats)

    def get_aggregate_non_norm_stats(self, abs_val=False, non_normed=True):
        return pickle.loads(self.calc_aggregate_stats(abs_val=abs_val, non_normed=non_normed))

    def view_3d(self):
        from importlib import import_module
        ShapeRotator = import_module("viewers.shape_rotator").ShapeRotator
        ShapeRotator.plot_3d_sg(self)


        
    # def get_tsfresh_data(self):
    #     matrix_df = self.mat_df()
    #     matrix_df['id'] = 0
    #     features = extract_features(matrix_df, column_id='id', column_sort='time')
    #     # print(features.describe())
