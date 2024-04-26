import csv
from datetime import datetime
from importlib import import_module
import plotly.offline as py
import os
from types import NoneType
from typing import List
import sqlite3
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient_task import PatientTask
from models.legacy_trial import Trial
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import plotly.express as px
from scipy.stats import t
import plotly.io as pio
from models.legacy_cohort import Cohort

from viewers.multi_plotter import MultiPlotter
from viewers.plotter import Plotter


# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.

class Task(LegacyBaseModel):
    table_name = "task"

    def __init__(self, id=None, description=None, created_at=None, updated_at=None, is_dominant=False):
        super().__init__()
        self.id = id
        self.description = description
        self.created_at = created_at
        self.updated_at = updated_at
        self.is_dominant = is_dominant

    def get_patients(self):
        from models.legacy_patient import Patient
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
        Trial = import_module("models.legacy_trial").Trial
        self._cursor.execute("""
            SELECT trial.* FROM trial
            JOIN patient_task ON trial.patient_task_id = patient_task.id
            WHERE patient_task.task_id = ?
        """, (self.id,))

        return [Trial(*row) for row in self._cursor.fetchall()]


    def get_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        GradientSet = import_module("models.legacy_gradient_set").GradientSet
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


    def get_position_sets_for_sensor(self, sensor):
        from importlib import import_module
        PositionSet = import_module("models.legacy_position_set").PositionSet
        query = f"""
            SELECT position_set.*
            FROM position_set
            INNER JOIN trial ON trial.id = position_set.trial_id
            INNER JOIN patient_task ON patient_task.id = trial.patient_task_id
            WHERE patient_task.task_id = ? AND position_set.sensor_id = ?
        """

        self._cursor.execute(query, (self.id, sensor.id))
        position_sets = [PositionSet(*row) for row in self._cursor.fetchall()]
        return position_sets

    def get_pos_sets(self):
        from importlib import import_module
        PositionSet = import_module("models.legacy_position_set").PositionSet
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
        GradientSet = import_module("models.legacy_gradient_set").GradientSet
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

    def box_whisker_plot(self, dataframes: List[pd.DataFrame]):
        data = [df.iloc[:, 1] for df in dataframes]
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xlabel('Dataframes')
        ax.set_ylabel('Values')
        plt.show()

    def combined_gradient_set_stats_by_task(self, sensor, loc='grad_data__sum_values'):
        print("yolo 1")
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = []
        for gradient_set in gradient_sets:
            if gradient_set.get_aggregate_stats() is not None:
                print(f"yolo {gradient_set.id}")
                aggregated_stats = gradient_set.get_aggregate_stats().loc[loc]

                
                plotter = Plotter(aggregated_stats)
                plotters.append(plotter)
        multi_plotter = MultiPlotter(plotters)
        combined_stats_series = multi_plotter.combined_stats()
        ns = []
        ns.append(Plotter(combined_stats_series))
        return MultiPlotter(ns).display_combined_box_plot(title=f"Task {self.id}: {self.description}, Sensor: {sensor.name}, TS index: {loc}")

    def get_counterpart_task(self, alt=None):
        """
        If the current task's description contains 'dominant', 
        this function returns the counterpart 'non-dominant' task.
        """
        # if we are passing a custom cohort the counterpart task will be from a counterpart cohort.
        if 'nondominant' in self.description.lower():
            # Replace this with the actual query to fetch the dominant task
            
            dominant_description = self.description.replace('nondominant', 'dominant').capitalize()
            return Task.where(description=dominant_description)
        elif 'dominant' in self.description.lower():
            # Replace this with the actual query to fetch the non-dominant task
            # I'm assuming you have some sort of method to find tasks by patient ID and description
            non_dominant_description = self.description.replace('dominant', 'nondominant').capitalize()
            return Task.where(description=non_dominant_description)



        else:
            print("This task's description does not contain 'dominant' or 'nondominant'")
            return None

    @classmethod
    def dominant(cls):
        cls._cursor.execute("""
            SELECT * FROM task
            WHERE description LIKE '%dominant%'
            AND description NOT LIKE '%nondominant%'
        """)
        return [cls(*row) for row in cls._cursor.fetchall()]

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

    def combined_gradient_set_stats_by_patient(self, sensor, loc='grad_data__sum_values'):
        from importlib import import_module
        PatientTask = import_module("models.legacy_patient_task").PatientTask
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
        gradient_sets = self.get_gradient_sets_for_sensor(sensor)
        plotters = [Plotter(gs.get_aggregate_stats().loc[loc]) for gs in gradient_sets 
                    if gs.aggregated_stats is not None and gs.get_aggregate_stats().loc[loc] is not None]
        multi_plotter = MultiPlotter(plotters)
        labels = [f"{gs.get_patient().name}" for gs in gradient_sets if gs.aggregated_stats is not None]
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
        ignore = [
            'grad_data__has_duplicate_max',
            'grad_data__has_duplicate_min',
            'grad_data__has_duplicate',
            'grad_data__sum_values',
        ]
        statistics_list = [
            'grad_data__mean',
            'grad_data__length',
            'grad_data__abs_energy',
            'grad_data__mean_abs_change',
            'grad_data__mean_change',
            'grad_data__mean_second_derivative_central',
            'grad_data__median',
            'grad_data__standard_deviation',
            'grad_data__variation_coefficient',
            'grad_data__variance',
            'grad_data__skewness',
            'grad_data__kurtosis',
            'grad_data__root_mean_square',
            'grad_data__absolute_sum_of_changes',
            'grad_data__longest_strike_below_mean',
            'grad_data__longest_strike_above_mean',
            'grad_data__count_above_mean',
            'grad_data__count_below_mean',
            'grad_data__last_location_of_maximum',
            'grad_data__first_location_of_maximum',
            'grad_data__last_location_of_minimum',
            'grad_data__first_location_of_minimum',
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
        Sensor = import_module("models.legacy_sensor").Sensor

        sensors = [
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "rfrm_x",
            "rfrm_y",
            "rfrm_z",
            "relb_x",
            "relbm_x",
            "relb_y",
            "relbm_y",
            "relb_z",
            "relbm_z",
            "rupa_x",
            "rupa_y",
            "rupa_z",
            "rsho_x",
            "rsho_y",
            "rsho_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            'rfhd_x', 
            'rfhd_y', 
            'rfhd_z',  
            'rbhd_x', 
            'rbhd_y', 
            'rbhd_z',
            'rfin_x', 
            'rfin_y', 
            'rfin_z',
        ]
        
        sensors = Sensor.where(name=names)

        for stat in statistics_list:
            print(f"Processing: {stat}")
            plot = self.plot_combined_stats(sensors, loc=stat)

            # You can change this filename format as you need
            filename = os.path.join(full_dir_path, f"BoxPlot_{self.description}_{stat}.html")

            # Save the plot as an HTML file
            py.plot(plot, filename=filename, auto_open=False)
        print("done!")

    def sensor_translation(self, sensor_name, min=False, axis=False):
        translation_dict = {
            'l': 'Left',
            'r': 'Right',
            'wra': 'Wrist Sensor A',
            'wrb': 'Wrist Sensor B',
            'frm': 'Forearm',
            'bhd': 'Back of Hand',
            'fin': 'Finger',
            'sho': 'Shoulder',
            'elb': 'Elbow',
            'elbm': 'Elbow Motion',
            'upa': 'Upper Arm',
            'x': 'x axis',
            'y': 'y axis',
            'z': 'z axis',
        }

        # Split the sensor name into parts
        side = sensor_name[0]
        sensor = sensor_name[1:5] if sensor_name[1:5] in translation_dict else sensor_name[1:4]
        axis = sensor_name[-1]

        # Get the full names from the dictionary
        side_full = translation_dict.get(side)
        sensor_full = translation_dict.get(sensor)
        axis_full = translation_dict.get(axis)

        # Return the full sensor name
        if not min:
            return f"{side_full} {sensor_full}"

        if axis:
            return axis_full

        return f"{side_full} {sensor_full} {axis_full}"


    def generate_parallel_coordinates(self, self_means, counterpart_means, t_stat, p_score, sensor_name, description, loc, directory='parallel_plots_abs/'):
        import plotly.graph_objects as go

        if len(self_means) != len(counterpart_means):
            print("NOooooo!")
            return
        opposite_sensor_name = self.get_counterpart_sensor(sensor_name)
        sensor_string = self.sensor_translation(sensor_name)
        opposite_sensor_string = self.sensor_translation(opposite_sensor_name)
        description_string = description.split("_")[0]
        base_name = self.sensor_translation(sensor_name, min=True)
        axis_name = self.sensor_translation(sensor_name, axis=True)
        # Create the Plotly figure
        fig = go.Figure()

        # Add traces for each measurement
        for self_mean, counterpart_mean in zip(self_means, counterpart_means):
            fig.add_trace(
                go.Scatter(
                    x=[f"Dominant", f"Non Dominant"],
                    y=[self_mean[1], counterpart_mean[1]],
                    mode='lines+markers+text',
                    name=self_mean[0], # Name of the trace will be the patient id
                    showlegend=False,  # hide the legend
                    hovertemplate='Patient ID: %{name}<br>Mean: %{y}'
                )
            )
        
        if description_string == "Block":
            size = [.4,.64]
        elif description_string == "Rings":
            size = [.4,.64]
        elif description_string == "Switchside":
            size = [.4,.64]
        else:
            size = []
        # Set labels and title
        if len(size) != 0:
            fig.update_layout(
                title=
                    f'Task: {str(description_string)}, Feature: {loc}<br>' +
                    f'{str(axis_name)}<br>' +
                    f'{str(base_name)}<br>' +
                    f't-stat: {t_stat:.2f}, p-score {p_score}<br>',
                width=950,
                height=1500,
                margin=dict(t=240),
                font=dict(    # Increase the font size
                    size=18,
                )
            )
        else:
            fig.update_layout(
                title=
                    f'Task: {str(description_string)}, Feature: {loc}<br>' +
                    f'{str(axis_name)}<br>' +
                    f'{str(base_name)}<br> ' +
                    f't-stat: {t_stat:.2f}, p-score {p_score}<br>',
                width=950,
                height=1500,
                margin=dict(t=240),
                font=dict(    # Increase the font size
                    size=18,
                )
            )



        # Save the figure as HTML
        if not os.path.exists(directory):
            os.makedirs(directory)
        pio.write_image(fig, f'{directory}/{str(description_string)}_{str(sensor_name)}.png')

    def get_counterpart_sensor(self, sensor_name):
        if sensor_name.startswith('l'):
            return 'r' + sensor_name[1:]
        elif sensor_name.startswith('r'):
            return 'l' + sensor_name[1:]
        else:
            return None

    @classmethod
    def get_counterpart_sensor(cls, sensor_name):
        if sensor_name.startswith('l'):
            return 'r' + sensor_name[1:]
        elif sensor_name.startswith('r'):
            return 'l' + sensor_name[1:]
        else:
            return None

    def dom_nondom_stats(self, loc='grad_data__abs_energy', abs_val=False, non_normed=False, dynamic=False, cohort=None):
        Sensor = import_module("models.legacy_sensor").Sensor
        Patient = import_module("models.legacy_patient").Patient

        names = [
            "rwra_x",
            "lwra_x",
            "rwrb_x",
            "lwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "rfrm_x",
            "rfrm_y",
            "rfrm_z",
            "relb_x",
            "relbm_x",
            "relb_y",
            "relbm_y",
            "relb_z",
            "relbm_z",
            "rupa_x",
            "rupa_y",
            "rupa_z",
            "rsho_x",
            "rsho_y",
            "rsho_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            'rfhd_x', 
            'rfhd_y', 
            'rfhd_z',  
            'rbhd_x', 
            'rbhd_y', 
            'rbhd_z',
            'rfin_x', 
            'rfin_y', 
            'rfin_z',
        ]

        all_sensors = Sensor.where(name=names)  # get all sensors with names in the `names` list
        sensors = sorted(all_sensors, key=lambda sensor: names.index(sensor.name))
        if cohort is not None:
            self_pts = PatientTask.where(task_id=self.id, cohort_id=cohort.id)   
        else:
            self_pts = PatientTask.where(task_id=self.id)
        row = []
        p_row = []

        for sensor in sensors:
            if sensor.side != 'right':
                continue

            self_means = []
            counterpart_means = []

            counterpart_tasks = self.get_counterpart_task()
            if not counterpart_tasks:
                print(f"No counterpart task found for task {self.description} with id {self.id}")
                continue

            counterpart_task = counterpart_tasks[0]
            pts = len(self_pts)
            bads = 0
            for self_pt in self_pts:
                # This patient is a lefty skip them for now...
                # TODO: Stephen, update this to work w/lefties.
                if self_pt.patient_id == 5:
                    continue

                try:
                    counterpart_pts = PatientTask.where(task_id=counterpart_task.id, patient_id=self_pt.patient_id)
                    counterpart_sensor = Sensor.where(name=self.get_counterpart_sensor(sensor.name))[0]
                    if not counterpart_pts:
                        print(f"No counterpart patient task found for task with id {counterpart_task.id} and patient id {self_pt.patient_id}")
                        continue

                    counterpart_pt = counterpart_pts[0]
                    curr_patient = Patient.where(id=self_pt.patient_id)[0].name
                    if abs_val is True:
                        self_means.append([curr_patient, abs(self_pt.combined_gradient_set_stats(sensor, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, loc=loc)['mean'])])
                        counterpart_means.append([curr_patient, abs(counterpart_pt.combined_gradient_set_stats(counterpart_sensor, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, loc=loc)['mean'])])
                    else:
                        self_means.append([curr_patient, self_pt.combined_gradient_set_stats(sensor, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, loc=loc)['mean']])
                        counterpart_means.append([curr_patient, counterpart_pt.combined_gradient_set_stats(counterpart_sensor, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, loc=loc)['mean']])
                except TypeError as e:
                    print("No gradient sets found!")
                    return [[0.0], [0.0]]
                except KeyError as e:
                    bads += 1
                    print(f"KeyError {e} self pt {self_pt.id} counter {counterpart_pt.id}")
                    continue

            self_means_values = [mean[1] for mean in self_means]
            counterpart_means_values = [mean[1] for mean in counterpart_means]
            print("Task:", self.description, ", Counterpart Task:", counterpart_task.description)
            print("Sensor:", sensor.name, ", Counterpart Sensor:", counterpart_sensor.name)
            print("Bad:", bads, pts)

            try:
                t_stat, p_score = stats.ttest_ind(self_means_values, counterpart_means_values)
            except ValueError:
                t_stat, p_score = stats.ttest_rel(self_means_values, counterpart_means_values)

            print(f"t_stat: {t_stat} sensor = {sensor.name}, desc: {self.description}"),
            print(f"p_score: {p_score} sensor = {sensor.name}, desc: {self.description}")
            print("sensor_name:", sensor.name, "sensor_side", sensor.side)
            print("generating parallel plots...")

            addn = ""
            if abs_val is True:
                addn = addn + "abs_val"
            if non_normed is True:
                addn = addn+ "_non_normed"

            directory_path = f'parallel_plots_{addn}/{loc}/'
            directory_path = "generated_pngs_" + directory_path
            if sensor.name == 'lwra_x' or sensor.name == 'rwra_x' or sensor.name == 'lwrb_x' or sensor.name == 'rwrb_x':
                self.generate_parallel_coordinates(
                    self_means, counterpart_means, t_stat, p_score,
                    sensor.name, self.description, loc,
                    directory=directory_path
                )

            row.append(t_stat)
            p_row.append(p_score)

        return [row, p_row]



    @classmethod
    def generate_t_test_csv(cls, output_csv_path, output_csv_p_score_path, abs_val=False, non_normed=False, dynamic=False, cohort=None, loc='grad_data__abs_energy'):
        # Fetch all tasks
        # all_tasks = [task for task in cls.all() if ('dominant' in task.description and 'nondominant' not in task.description)]
        all_tasks = [task for task in cls.all() if ('nondominant' in task.description)]

        # Prepare to write to CSV
        with open(output_csv_path, 'w', newline='') as csvfile:
            with open(output_csv_p_score_path, 'w', newline='') as p_csvfile:
                writer = csv.writer(csvfile)
                p_writer = csv.writer(p_csvfile)

                # Write header row
                sensor_names = [
                    "rwra_x",
                    "lwra_x",
                    "rwrb_x",
                    "lwrb_x",
                    "rwra_y",
                    "rwrb_y",
                    "rwra_z",
                    "rwrb_z",
                    "rfrm_x",
                    "rfrm_y",
                    "rfrm_z",
                    "relb_x",
                    "relbm_x",
                    "relb_y",
                    "relbm_y",
                    "relb_z",
                    "relbm_z",
                    "rupa_x",
                    "rupa_y",
                    "rupa_z",
                    "rsho_x",
                    "rsho_y",
                    "rsho_z",
                    "rwra_x",
                    "rwrb_x",
                    "rwra_y",
                    "rwrb_y",
                    "rwra_z",
                    "rwrb_z",
                    'rfhd_x', 
                    'rfhd_y', 
                    'rfhd_z',  
                    'rbhd_x', 
                    'rbhd_y', 
                    'rbhd_z',
                    'rfin_x', 
                    'rfin_y', 
                    'rfin_z',
                ]
                writer.writerow(['Task ID t test', 'Task Description'] + sensor_names)
                p_writer.writerow(['Task ID p scores', 'Task Description'] + sensor_names)

                # Iterate over each task and generate t-test stats
                for task in all_tasks:
                    print(task.id)
                    if task.id == 4:
                        print("hi")
                        result = task.dom_nondom_stats(loc, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, cohort=cohort)
                        if result is not None:
                            t_test_stats, p_score_stats = result
                        else:
                            t_test_stats, p_score_stats = [], []

                        writer.writerow([task.id, task.description] + t_test_stats)
                        p_writer.writerow([task.id, task.description] + p_score_stats)

        print(f'T-test results saved to {output_csv_path}.')
        print(f'P score results saved to {output_csv_p_score_path}.')

    @classmethod
    def gen_all_stats_csv(cls, abs_val=False, non_normed=False, dynamic=False, cohort=None):
        ignore = [
            'grad_data__has_duplicate_max',
            'grad_data__has_duplicate_min',
            'grad_data__has_duplicate',
            'grad_data__sum_values',
            'grad_data__variance_larger_than_standard_deviation',
            
        ]
        statistics_list = [
            'grad_data__mean',
            'grad_data__length',
            'grad_data__abs_energy',
            'grad_data__mean_abs_change',
            'grad_data__mean_change',
            'grad_data__mean_second_derivative_central',
            'grad_data__median',
            'grad_data__standard_deviation',
            'grad_data__variation_coefficient',
            'grad_data__variance',
            'grad_data__skewness',
            'grad_data__kurtosis',
            'grad_data__root_mean_square',
            'grad_data__absolute_sum_of_changes',
            'grad_data__longest_strike_below_mean',
            'grad_data__longest_strike_above_mean',
            'grad_data__count_above_mean',
            'grad_data__count_below_mean',
            'grad_data__last_location_of_maximum',
            'grad_data__first_location_of_maximum',
            'grad_data__last_location_of_minimum',
            'grad_data__first_location_of_minimum',
        ]

        # Define the directory where the files will be saved
        if abs_val is True:
            directory = "generated_csvs/stats_list_abs"
        else:
            directory = "generated_csvs/stats_list"

        # Ensure the directory exists
        os.makedirs(directory, exist_ok=True)
        addn = ""
        if abs_val is True:
            addn = addn + "_abs_val"
        if non_normed is True:
            addn = addn+ "_non_normed"

        for stat in statistics_list:
            # Join the directory name with the file name
            path = os.path.join(directory, stat + addn + '.csv')
            p_path = os.path.join(directory, stat + addn + '_p_score' + '.csv')
            cls.generate_t_test_csv(path, p_path, abs_val=abs_val, non_normed=non_normed, dynamic=dynamic, cohort=cohort, loc=stat)
