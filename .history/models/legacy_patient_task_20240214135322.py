from typing import List
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import sqlite3
from models.legacy_patient import Patient
from viewers.multi_plotter import MultiPlotter
import pandas as pd
from viewers.plotter import Plotter


class PatientTask(LegacyBaseModel):
    table_name = "patient_task"

    def __init__(self, id=None, patient_id=None, task_id=None, created_at=None, updated_at=None, cohort_id=None):
        self.id = id
        self.patient_id = patient_id
        self.task_id = task_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.cohort_id = cohort_id

    def create(self, **kwargs):
        self.patient_id = kwargs.get("patient_id")
        self.task_id = kwargs.get("task_id")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_id = kwargs.get("patient_id", self.patient_id)
        self.task_id = kwargs.get("task_id", self.task_id)
        return super().update(**kwargs)

    def add_trial(self, trial):
        """
        Add a Trial to the PatientTask instance.

        :param trial: The Trial instance to be added.
        :return: None
        """
        if not hasattr(self, 'trials'):
            self.trials = []

        self.trials.append(trial)

        trial.patient_task_id = self.id
        trial.update(patient_task_id=self.id)
    
    def get_trials(self):
        from importlib import import_module
        Trial = import_module("models.legacy_trial").Trial
        return Trial.where(patient_task_id=self.id)

    def get_gradient_sets_for_sensor(self, sensor, all=False):
        from importlib import import_module
        GradientSet = import_module("models.legacy_gradient_set").GradientSet
        gradient_sets = []
        if all is True:
            print("all")
            Sensor = import_module("models.legacy_sensor").Sensor
            same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)
            gradient_sets = []

            for sensor_obj in same_sensor_ids:
                self._cursor.execute("""
                    SELECT gradient_set.*
                    FROM gradient_set
                    JOIN trial ON gradient_set.trial_id = trial.id
                    WHERE trial.patient_task_id = ? AND gradient_set.sensor_id = ?
                """, (self.id, sensor_obj.id))  # use sensor_obj.id instead of sensor_id
        
            gradient_set_rows = self._cursor.fetchall()
            gradient_sets.extend([GradientSet(*row) for row in gradient_set_rows])
        else:
            self._cursor.execute("""
                SELECT gradient_set.*
                FROM gradient_set
                JOIN trial ON gradient_set.trial_id = trial.id
                WHERE trial.patient_task_id = ? AND gradient_set.sensor_id = ?
            """, (self.id, sensor.id))

            gradient_set_rows = self._cursor.fetchall()
            gradient_sets = [GradientSet(*row) for row in gradient_set_rows]

        return gradient_sets

    def get_dynamic_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        DynamicGradientSet = import_module("models.gradient_set").DynamicGradientSet
        self._cursor.execute("""
            SELECT dynamic_gradient_set.*
            FROM dynamic_gradient_set
            JOIN trial ON dynamic_gradient_set.trial_id = trial.id
            WHERE trial.patient_task_id = ? AND dynamic_gradient_set.sensor_id = ?
        """, (self.id, sensor.id))

        dynamic_gradient_set_rows = self._cursor.fetchall()
        dynamic_gradient_sets = [DynamicGradientSet(*row) for row in dynamic_gradient_set_rows]

        return dynamic_gradient_sets

    def combined_gradient_set_stats(self, sensor, abs_val=False, non_normed=False, dynamic=False, loc='grad_data__sum_values'):
        from importlib import import_module
        Task = import_module("models.legacy_task").Task

        if dynamic is True:
            gradient_sets = self.get_dynamic_gradient_sets_for_sensor(sensor)    
        else:
            gradient_sets = self.get_gradient_sets_for_sensor(sensor)

    
        plotters = []
        for gradient_set in gradient_sets:
            if gradient_set.aggregated_stats is not None:
                if (not non_normed) and (not abs_val):
                    aggregated_stats = gradient_set.get_aggregate_stats()
                    if loc:
                        aggregated_stats = aggregated_stats.loc[loc]
                else:
                    if type(loc) is not list and loc:
                        loc = loc.replace('grad_data', gradient_set.name)
                    aggregated_stats = gradient_set.get_aggregate_non_norm_stats(abs_val=abs_val)
                    if loc:
                        aggregated_stats = aggregated_stats.loc[loc]
                plotter = Plotter(aggregated_stats)
                plotters.append(plotter)
        multi_plotter = MultiPlotter(plotters)
        combined_stats_series = multi_plotter.combined_stats()

        return combined_stats_series


    def combined_gradient_set_count(self, sensor, abs_val=False, non_normed=False, loc=['grad_data__sum_values']):
        gradient_sets = self.get_gradient_sets_for_sensor(sensor, all=False)

        # Check if there are gradient sets available
        if len(gradient_sets) == 0:
            print("empty", sensor.name, "patient_id:", self.patient_id, "task_id:", self.task_id)
            return []

        sub_gradient_lengths = []
        for gradient_set in gradient_sets:
            sg_len = len(gradient_set.sub_gradients())
            sub_gradient_lengths.append(sg_len)
        
        # Calculate the average length of subgradients if there are any lengths recorded
        if sub_gradient_lengths:
            avg_sg_len = sum(sub_gradient_lengths) / len(sub_gradient_lengths)
        else:
            avg_sg_len = 0

        # Prepare the result in the desired data structure
        result_df = pd.DataFrame({'grad_data__mean_sub_gradient_length': [avg_sg_len]})
    
        
        return result_df


    def combined_gradient_set_stats_list(self, sensor, abs_val=False, non_normed=False, dynamic=False, all=False, loc=['grad_data__sum_values']):
        sensors = sensor.get_set()
        if all is True:
            gradient_sets = []
            for sen in sensors:
                gradient_sets += self.get_gradient_sets_for_sensor(sen, all=False)
        else:
            gradient_sets = self.get_gradient_sets_for_sensor(sensor, all=False)
        if len(gradient_sets) == 0:
            print("empty", sensor.name, "patient_id:", self.patient_id, "task_id:", self.task_id)
            return []
    
        dataframes = []
        print("GETTING STATS FOR GRAD SETS")
        for gradient_set in gradient_sets:
            if gradient_set.aggregated_stats is not None:
                if not non_normed:
                    aggregated_stats = gradient_set.get_aggregate_stats()
                    if loc:
                        aggregated_stats = aggregated_stats.loc[loc]
                    dataframes.append(aggregated_stats)
                else:
                    aggregated_stats = gradient_set.get_aggregate_non_norm_stats(abs_val=abs_val, non_normed=non_normed)
                    if loc:
                        aggregated_stats = aggregated_stats.loc[loc]
                    dataframes.append(aggregated_stats)

        # Concatenate the list of dataframes vertically
        
        # Group by index and calculate the mean for each key
        try:
            combined_df = pd.concat(dataframes)
            mean_df = combined_df.groupby(combined_df.index).mean()
        except ValueError as e:
            import pdb;pdb.set_trace()

            
        return mean_df

    def get_patient(self):
        self._cursor.execute(f"SELECT * FROM patient WHERE id=?", (self.patient_id,))
        row = self._cursor.fetchone()
        if row:
            return Patient(*row)
        else:
            raise ValueError(f"Patient not found for PatientTask id {self.id}")

    @classmethod
    def get(cls, patient, task):
        cls._cursor.execute("SELECT * FROM patient_task WHERE patient_id=? AND task_id=?", (patient.id, task.id))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return None

    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()