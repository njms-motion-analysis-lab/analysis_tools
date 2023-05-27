from models.base_model import BaseModel
import sqlite3
from models.patient import Patient
from multi_plotter import MultiPlotter
import pandas as pd
from plotter import Plotter


class PatientTask(BaseModel):
    table_name = "patient_task"

    def __init__(self, id=None, patient_id=None, task_id=None, created_at=None, updated_at=None):
        self.id = id
        self.patient_id = patient_id
        self.task_id = task_id
        self.created_at = created_at
        self.updated_at = updated_at

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

    def get_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        self._cursor.execute("""
            SELECT gradient_set.*
            FROM gradient_set
            JOIN trial ON gradient_set.trial_id = trial.id
            WHERE trial.patient_task_id = ? AND gradient_set.sensor_id = ?
        """, (self.id, sensor.id))

        gradient_set_rows = self._cursor.fetchall()
        gradient_sets = [GradientSet(*row) for row in gradient_set_rows]

        return gradient_sets

    def combined_gradient_set_stats(self, sensor, loc='grad_data__sum_values'):
        from importlib import import_module
        Task = import_module("models.task").Task
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = []
        for gradient_set in gradient_sets:
            if gradient_set.aggregated_stats is not None:
                aggregated_stats = gradient_set.get_aggregate_stats().loc[loc]
                plotter = Plotter(aggregated_stats)
                plotters.append(plotter)
        multi_plotter = MultiPlotter(plotters)
        combined_stats_series = multi_plotter.combined_stats()
        
        return combined_stats_series

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