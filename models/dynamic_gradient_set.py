from models.dynamic_position_set import DynamicPositionSet
import pickle
from typing import Any, List
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient import Patient
from datetime import datetime
from models.dynamic_sub_gradient import DynamicSubGradient
from tsfresh import extract_features
import pandas as pd
import numpy as np
import pdb


from models.legacy_patient_task import PatientTask
from exp_motion_sample_trial import ExpMotionSampleTrial
from motion_filter import MotionFilter

class DynamicGradientSet(LegacyBaseModel):
    table_name = "dynamic_gradient_set"

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
        print("yo")
        if not self.sensor_id:
            return None
        from importlib import import_module
        Sensor = import_module("models.sensor").Sensor

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
        from importlib import import_module
        Task = import_module("models.task").Task
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

    def get_dynamic_position_set(self):
        return DynamicPositionSet.where(trial_id=self.trial_id, sensor_id=self.sensor_id)[0]

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
            print("This DynamicGradientSet is already associated with the provided sensor.")
            return

        self.sensor_id = sensor.id
        self.update(sensor_id=self.sensor_id)
        print(f"Sensor with ID {sensor.id} has been associated with this DynamicGradientSet.")

    def get_patient_task_id(self):
        self._cursor.execute("SELECT patient_task_id FROM dynamic_gradient_set WHERE id=?", (self.id,))
        return self._cursor.fetchone()[0]

    def get_patient_task(self):
        patient_task_id = self.get_patient_task_id()
        return PatientTask.get(patient_task_id)


    def create_dynamic_subgradients(self):
        print("CREATING DYNAMIC SUBS")
        matrix = self.mat()
        dynamic_sub_gradients = []
        start_time = 0
        position_set = DynamicPositionSet.where(name=self.name,trial_id=self.trial_id,sensor_id=self.sensor_id)
        print(position_set)

        p_matrix = DynamicPositionSet.where(name=self.name,trial_id=self.trial_id,sensor_id=self.sensor_id)[0].mat()

        for i in range(1, len(matrix)):
            if (matrix.iloc[i] >= 0 and matrix.iloc[i - 1] < 0) or (matrix.iloc[i] <= 0 and matrix.iloc[i - 1] > 0):
                stop_time = i - 1
                current_slice = matrix.loc[matrix.index[start_time]:matrix.index[stop_time]]
                p_slice = p_matrix.loc[matrix.index[start_time]:matrix.index[stop_time]]
                valid = self.is_valid(current_slice, p_slice)
                normalized = DynamicSubGradient.normalize(current_slice),
                if len(DynamicSubGradient.where(dynamic_gradient_set_id=self.id, dynamic_gradient_set_ord=len(dynamic_sub_gradients), name=self.name)) != 0:
                    
                    
                    print("dsg already exists...")
                    print("matrix:", matrix)
                    print("slice", current_slice)
                    print("normalized:", normalized)
                    print("XXX")
                    
                else:
                    print("matrix:", matrix)
                    print("slice", current_slice)
                    print("normalized:", normalized)
                    print("YYY")
                    subgradient = DynamicSubGradient.find_or_create(
                        name=self.name,
                        valid=valid,
                        matrix=current_slice,
                        dynamic_gradient_set_id=self.id,
                        dynamic_gradient_set_ord=len(dynamic_sub_gradients),
                        start_time=matrix.index[start_time],
                        stop_time=matrix.index[stop_time],
                        mean=current_slice.mean(),
                        median=current_slice.median(),
                        stdev=current_slice.std(),
                        normalized = DynamicSubGradient.normalize(current_slice),
                    )

                    ts_stats = DynamicSubGradient.get_tsfresh_stats(subgradient)
                    print("ts_stats:", ts_stats)

                    non_normalized_ts_stats = DynamicSubGradient.get_tsfresh_stats_non_normalized(subgradient)
                    print("non_norm_ts:", non_normalized_ts_stats)
                    pos_ts_stats = DynamicSubGradient.get_tsfresh_stats_position(subgradient, manual_position_data=p_slice)
                    print("pos_ts:", pos_ts_stats)
                    subgradient.update(
                        submovement_stats=ts_stats,
                        submovement_stats_nonnorm=non_normalized_ts_stats,
                        submovement_stats_position=pos_ts_stats
                    )
                    print("sg_sms", subgradient.submovement_stats)
                    print("sg_id", subgradient.id)
                    print("sg_created", subgradient.created_at)


                    print("yolo 5")
                    dynamic_sub_gradients.append(subgradient)
                    start_time = i
        # print("created subgrads")
        return dynamic_sub_gradients

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
    
    def dynamic_sub_gradients(self):
        from importlib import import_module
        DynamicSubGradient = import_module("models.dynamic_sub_gradient").DynamicSubGradient

        return DynamicSubGradient.where(dynamic_gradient_set_id=self.id)

    @classmethod
    def create_all_available_sub_gradients(cls):
        all_gs = DynamicGradientSet.all()
        created_sg = []
        # print("number of gradient sets receiving dynamic_sub_gradients:", len(all_gs))
        i = 0
        for gs in all_gs:
            # print("creating dynamic_sub_gradients for:", gs.name, gs.sensor_id, gs.trial_id)
            sg = gs.create_dynamic_subgradients()
            i += 1
            created_sg += sg
            # print("created dynamic_sub_gradients for gs:", i)
        print("done")

        return created_sg

    def calc_aggregate_stats(self, abs=True, normalized=True):
        # print("AGGREGATOZIGIGIGNG")
        #get each submovement's stats
        sub_stats_all = pd.DataFrame()
        subgrads = self.dynamic_sub_gradients()
        for subgrad in subgrads:
            if subgrad.valid:
                sub_stats = subgrad.get_sub_stats(normalized=normalized)
                sub_stats_all = sub_stats_all.append(sub_stats)
                #sub_stats_all = pd.concat([sub_stats_all, pd.DataFrame([sub_stats])], ignore_index=True)

        # for each stat type in the submovement stats, calculate aggregate stat for the whole trial
        stats = pd.DataFrame()
        for colname, colvalues in sub_stats_all.iteritems():
            # print(colname)
            if abs is True:
                colvalues = np.abs(colvalues)

            aggregate = {"mean": np.mean(colvalues), "median": np.median(colvalues), 
                            "sd":np.std(colvalues), "IQR": np.subtract(*np.percentile(colvalues, [75, 25])),
                            "10th": np.percentile(colvalues, 10), "90th": np.percentile(colvalues, 90)}
            #print(aggregate_stats)
            stats = pd.concat([stats, pd.DataFrame([aggregate], index=[colname])])
        #print(stats)
        #self.aggregate = memoryview(pickle.dumps(motionstats))
        return memoryview(pickle.dumps(stats))

    def get_aggregate_stats(self):
        return pickle.loads(self.aggregated_stats)

    def get_aggregate_non_norm_stats(self):
        return pickle.loads(self.calc_aggregate_stats(normalized=False))
        
    # def get_tsfresh_data(self):
    #     matrix_df = self.mat_df()
    #     matrix_df['id'] = 0
    #     features = extract_features(matrix_df, column_id='id', column_sort='time')
    #     # print(features.describe())
