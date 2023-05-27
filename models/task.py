import csv
from datetime import datetime
from importlib import import_module
import plotly.offline as py
import os
from types import NoneType
from typing import List
import sqlite3
from models.base_model import BaseModel
from models.patient_task import PatientTask
from models.trial import Trial
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

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
        print("yolo 1")
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = []
        for gradient_set in gradient_sets:
            if gradient_set.get_aggregate_stats() is not Empty:
                print(f"yolo {gradient_set.id}")
                aggregated_stats = gradient_set.get_aggregate_stats().loc[loc]

                
                plotter = Plotter(aggregated_stats)
                plotters.append(plotter)
        print("yolo 3")
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
    

    def plot_combined_stats(self, sensors, loc='grad_data__sum_values'):
        print("yolo 1")
        ns = []
        labels = []
        for sensor in sensors:
            gradient_sets = self.get_gradient_sets_for_sensor(sensor)
            plotters = []
            for gradient_set in gradient_sets:
                if not gradient_set.get_aggregate_stats().empty:
                    aggregated_stats = gradient_set.get_aggregate_stats().loc[loc]
                    plotter = Plotter(aggregated_stats)
                    plotters.append(plotter)
                else:
                    print(f"GS: {gradient_set.id} Aggregated stats DataFrame is empty, cannot access loc.")
            multi_plotter = MultiPlotter(plotters)
            combined_stats_series = multi_plotter.combined_stats()
            ns.append(Plotter(combined_stats_series))
            labels.append(sensor.name)
        return MultiPlotter(ns).create_combined_box_plot_py(labels=labels,title=f"Task: {self.description}, TS index: {loc},")

    def plot_and_save_all(self):
        statistics_list = [
            'grad_data__variance_larger_than_standard_deviation',
            'grad_data__has_duplicate_max',
            'grad_data__has_duplicate_min',
            'grad_data__has_duplicate',
            'grad_data__sum_values',
            'grad_data__abs_energy',
            'grad_data__mean_abs_change',
            'grad_data__mean_change',
            'grad_data__mean_second_derivative_central',
            'grad_data__median',
            'grad_data__mean',
            'grad_data__length',
            'grad_data__standard_deviation',
            'grad_data__variation_coefficient',
            'grad_data__variance',
            'grad_data__skewness',
            'grad_data__kurtosis',
            'grad_data__root_mean_square',
        ]

        # Get the current date and time
        now = datetime.now()
        datetime_str = now.strftime("%Y%m%d_%H%M%S")

        # Define the existing directory name
        existing_dir = "generated_plots"

        # Create a new directory under the existing directory
        dir_name = f"{self.description}_{datetime_str}_combined_plots"
        full_dir_path = os.path.join(existing_dir, dir_name)
        os.makedirs(full_dir_path, exist_ok=True)
        
        from importlib import import_module
        Sensor = import_module("models.sensor").Sensor
        names = ["lwra_x","lwrb_x","lwra_y","lwrb_y","lwra_z","lwrb_z","rwra_x","rwrb_x","rwra_y","rwrb_y","rwra_z","rwrb_z",]
        sensors = Sensor.where(name=names)

        for stat in statistics_list:
            print(f"Processing: {stat}")
            plot = self.plot_combined_stats(sensors, loc=stat)

            # You can change this filename format as you need
            filename = os.path.join(full_dir_path, f"BoxPlot_{self.description}_{stat}.html")

            # Save the plot as an HTML file
            py.plot(plot, filename=filename, auto_open=False)
        print("done!")

    
    def dom_nondom_stats(self, loc='grad_data__sum_values'):
        Sensor = import_module("models.sensor").Sensor
        names = ["lwra_x","lwrb_x","lwra_y","lwrb_y","lwra_z","lwrb_z","rwra_x","rwrb_x","rwra_y","rwrb_y","rwra_z","rwrb_z",]
        sensors = Sensor.where(name=names)
        self_pts = PatientTask.where(task_id=self.id)
        row = []
        for sensor in sensors:
            self_means = []
            counterpart_means = []
            counterpart_tasks = self.get_counterpart_task()
            if not counterpart_tasks:
                print(f"No counterpart task found for task {self.description} with id {self.id}")
                continue
            counterpart_task = counterpart_tasks[0]
            for self_pt in self_pts:
                try:
                    counterpart_pts = PatientTask.where(task_id=counterpart_task.id, patient_id=self_pt.patient_id)
                    if not counterpart_pts:
                        print(f"No counterpart patient task found for task with id {counterpart_task.id} and patient id {self_pt.patient_id}")
                        continue
                    counterpart_pt = counterpart_pts[0]
                    self_means.append(self_pt.combined_gradient_set_stats(sensor, loc=loc)['mean'])
                    counterpart_means.append(counterpart_pt.combined_gradient_set_stats(sensor, loc=loc)['mean'])
                except KeyError as e:
                    print(f"KeyError {e} self pt {self_pt.id} counter {counterpart_pt.id}")
                    continue

            t_stat, _ = stats.ttest_ind(self_means, counterpart_means)
            print(f"t_stat: {t_stat} sensor = {sensor.name}, desc: {self.description}")    
            row.append(t_stat)
        return row

    @classmethod
    def generate_t_test_csv(cls, output_csv_path, loc='grad_data__sum_values'):
        # Fetch all tasks
        all_tasks = [task for task in cls.all() if ('dominant' in task.description and 'nondominant' not in task.description)]



        # Prepare to write to CSV
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            sensor_names = ["lwra_x", "lwrb_x", "lwra_y", "lwrb_y", "lwra_z", "lwrb_z",
                            "rwra_x", "rwrb_x", "rwra_y", "rwrb_y", "rwra_z", "rwrb_z"]
            writer.writerow(['Task ID', 'Task Description'] + sensor_names)

            # Iterate over each task and generate t-test stats
            for task in all_tasks:
                t_test_stats = task.dom_nondom_stats(loc)
                writer.writerow([task.id, task.description] + t_test_stats)

        print(f'T-test results saved to {output_csv_path}.')

    @classmethod
    def gen_all_stats_csv(cls):
        statistics_list = [
            'grad_data__variance_larger_than_standard_deviation',
            'grad_data__has_duplicate_max',
            'grad_data__has_duplicate_min',
            'grad_data__has_duplicate',
            'grad_data__sum_values',
            'grad_data__abs_energy',
            'grad_data__mean_abs_change',
            'grad_data__mean_change',
            'grad_data__mean_second_derivative_central',
            'grad_data__median',
            'grad_data__mean',
            'grad_data__length',
            'grad_data__standard_deviation',
            'grad_data__variation_coefficient',
            'grad_data__variance',
            'grad_data__skewness',
            'grad_data__kurtosis',
            'grad_data__root_mean_square',
        ]

        # Define the directory where the files will be saved
        directory = "stats_list"

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)

        for stat in statistics_list:
            # Join the directory name with the file name
            path = os.path.join(directory, stat + '.csv')
            cls.generate_t_test_csv(path, loc=stat)

    def get_counterpart_task(self):
        """
        If the current task's description contains 'dominant', 
        this function returns the counterpart 'non-dominant' task.
        """
        if 'dominant' in self.description.lower():
            # Replace this with the actual query to fetch the non-dominant task
            # I'm assuming you have some sort of method to find tasks by patient ID and description
            non_dominant_description = self.description.replace('dominant', 'nondominant').capitalize()
            print(non_dominant_description)
            return Task.where(description=non_dominant_description)

        elif 'nondominant' in self.description.lower():
            # Replace this with the actual query to fetch the dominant task
            
            dominant_description = self.description.replace('nondominant', 'dominant').capitalize()
            return Task.where(description=dominant_description)

        else:
            print("This task's description does not contain 'dominant' or 'nondominant'")
            return None

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
