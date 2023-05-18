from types import NoneType
from typing import List
import sqlite3
from models.base_model import BaseModel
from models.patient_task import PatientTask
from models.trial import Trial
import pandas as pd
import matplotlib.pyplot as plt

from multi_plotter import MultiPlotter
from plotter import Plotter


# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.

class Task(BaseModel):
    table_name = "task"

    def __init__(self, id=None, description=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.description = description
        self.created_at = created_at
        self.updated_at = updated_at

    def get_patients(self):
        from models.patient import Patient
        self._cursor.execute("""
            SELECT patient.* FROM patient
            JOIN patient_task ON patient.id = patient_task.patient_id
            WHERE patient_task.task_id = ?
        """, (self.id,))
        return [Patient.get(row[0]) for row in self._cursor.fetchall()]

    def add_patient(self, patient):
        # Add a patient to the task.
        #x
        # Input:
        # - `patient`: the `Patient` object to associate with the task.
        #
        # Output:
        # - None.
        print(patient.id)
        self._cursor.execute("INSERT INTO patient_task (patient_id, task_id) VALUES (?, ?)", (patient.id, self.id))
        self._conn.commit()

    def trials(self):
        from importlib import import_module
        Trial = import_module("models.trial").Trial
        self._cursor.execute("""
            SELECT trial.* FROM trial
            JOIN patient_task ON trial.patient_task_id = patient_task.id
            WHERE patient_task.task_id = ?
        """, (self.id,))

        return [Trial(*row) for row in self._cursor.fetchall()]


    def get_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        query = f"""
            SELECT gradient_set.*
            FROM gradient_set
            INNER JOIN trial ON trial.id = gradient_set.trial_id
            INNER JOIN patient_task ON patient_task.id = trial.patient_task_id
            WHERE patient_task.task_id = ? AND gradient_set.sensor_id = ?
        """

        self._cursor.execute(query, (self.id, sensor.id))
        gradient_sets = [GradientSet(*row) for row in self._cursor.fetchall()]
        return gradient_sets
    
    def get_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        query = f"""
            SELECT gradient_set.*
            FROM gradient_set
            INNER JOIN trial ON trial.id = gradient_set.trial_id
            INNER JOIN patient_task ON patient_task.id = trial.patient_task_id
            WHERE patient_task.task_id = ? AND gradient_set.sensor_id = ?
        """

        self._cursor.execute(query, (self.id, sensor.id))
        gradient_sets = [GradientSet(*row) for row in self._cursor.fetchall()]
        return gradient_sets

    def get_pos_sets(self):
        from importlib import import_module
        PositionSet = import_module("models.position_set").PositionSet
        query = f"""
            SELECT position_set.*
            FROM position_set
            INNER JOIN trial ON trial.id = position_set.trial_id
            INNER JOIN patient_task ON patient_task.id = trial.patient_task_id
            WHERE patient_task.task_id = ?
        """

        self._cursor.execute(query, (self.id,))
        position_sets = [PositionSet(*row) for row in self._cursor.fetchall()]
        return position_sets

    def gradient_sets(self):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        query = f"""
            SELECT gradient_set.*
            FROM gradient_set
            INNER JOIN trial ON trial.id = gradient_set.trial_id
            INNER JOIN patient_task ON patient_task.id = trial.patient_task_id
            WHERE patient_task.task_id = ?
        """

        self._cursor.execute(query, (self.id,))
        gradient_sets = [GradientSet(*row) for row in self._cursor.fetchall()]
        return gradient_sets

    # TODO: Make this work.
    def box_whisker_plot(self, dataframes: List[pd.DataFrame]):
        # Collect data from each dataframe into a list
        data = [df.iloc[:, 1] for df in dataframes]  # Updated to use the second column (index 1)

        # Create a box and whisker plot
        fig, ax = plt.subplots()
        ax.boxplot(data)

        # Set axis labels
        ax.set_xlabel('Dataframes')
        ax.set_ylabel('Values')

        # Show the plot
        plt.show()

    def combined_gradient_set_stats_by_task(self, sensor, loc='grad_data__sum_values'):
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = []
        for gradient_set in gradient_sets:
            if gradient_set.aggregated_stats is not None:
                aggregated_stats = gradient_set.get_aggregate_stats().loc[loc]
                print(aggregated_stats)
                plotter = Plotter(aggregated_stats)
                plotters.append(plotter)
        print('task')
        multi_plotter = MultiPlotter(plotters)
        combined_stats_series = multi_plotter.combined_stats()
        ns = []
        ns.append(Plotter(combined_stats_series))
        return MultiPlotter(ns).display_combined_box_plot(title=f"Task {self.id}: {self.description}, Sensor: {sensor.name}, TS index: {loc}")


    def combined_gradient_set_stats_by_patient(self, sensor, loc='grad_data__sum_values'):
        from importlib import import_module
        PatientTask = import_module("models.patient_task").PatientTask
        patient_task_rows = self.get_patient_tasks()
        patient_tasks = [PatientTask(*row) for row in patient_task_rows]

        combined_stats_list = []
        labels = []

        for pt in patient_tasks:
            combined_stats = Plotter(pt.combined_gradient_set_stats(sensor, loc=loc))
            if combined_stats is not None:
                combined_stats_list.append(combined_stats)
                patient = pt.get_patient()
                labels.append(f"{patient.name}")

        if combined_stats_list:
            multi_plotter = MultiPlotter(combined_stats_list)
            multi_plotter.display_combined_box_plot(labels=labels, title=f"Task {self.id}: {self.description}, Sensor: {sensor.name}, TS index {loc}")
        else:
            print("No data available for the given sensor and task.")


    def combined_gradient_set_stats_by_patient_trial(self, sensor, loc='grad_data__sum_values'):
        print("hi")
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = []
        print("yo")
        i = 0
        for gs in gradient_sets:
            if gs.aggregated_stats is not None:
                mean_values = gs.get_aggregate_stats().loc[loc]
                if mean_values is not NoneType:
                    print(mean_values)
                    print(i)
                    i += 1
                    plotter = Plotter(mean_values)
                    print('next')
                    plotters.append(plotter)
                    print('next 2')
        print('done')
        multi_plotter = MultiPlotter(plotters)
        
        # Customize the labels with the patient name + the trial number
        # labels = [f"{gs.get_patient().name} Trial ID {gs.trial_id}" for gs in gradient_sets]
        labels = [
            f"{gs.get_patient().name}"
            for gs in gradient_sets
            if gs.aggregated_stats is not None
        ]
        multi_plotter.display_combined_box_plot(labels, title=f"Task {self.id}: {self.description}, Sensor: {sensor.name}, TS index {loc}")
    


    def get_patient_tasks(self):
        self._cursor.execute(f"SELECT * FROM patient_task WHERE task_id=?", (self.id,))
        rows = self._cursor.fetchall()
        return rows

    def get_pos_set_matrices(self):
        pos_sets = self.get_pos_sets()
        pos_set_matrices = [pos_set.get_matrix("matrix") for pos_set in pos_sets]
        return pos_set_matrices

    def __str__(self) -> str:
        return self.description
