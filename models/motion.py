from typing import List
import sqlite3
from models.base_model import BaseModel
from models.trial import Trial
import pandas as pd
import matplotlib.pyplot as plt


# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.

class Motion(BaseModel):
    table_name = "motion"

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
            JOIN patient_motion ON patient.id = patient_motion.patient_id
            WHERE patient_motion.motion_id = ?
        """, (self.id,))
        return [Patient.get(row[0]) for row in self._cursor.fetchall()]

    def add_patient(self, patient):
        # Add a patient to the motion.
        #x
        # Input:
        # - `patient`: the `Patient` object to associate with the motion.
        #
        # Output:
        # - None.
        print(patient.id)
        self._cursor.execute("INSERT INTO patient_motion (patient_id, motion_id) VALUES (?, ?)", (patient.id, self.id))
        self._conn.commit()

    def get_trials(self):
        from importlib import import_module
        Trial = import_module("models.trial").Trial
        self._cursor.execute("""
            SELECT trial.* FROM trial
            JOIN patient_motion ON trial.patient_motion_id = patient_motion.id
            WHERE patient_motion.motion_id = ?
        """, (self.id,))

        return [Trial(*row) for row in self._cursor.fetchall()]


    def get_gradient_sets_for_sensor(self, sensor):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        query = f"""
            SELECT gradient_set.*
            FROM gradient_set
            INNER JOIN trial ON trial.id = gradient_set.trial_id
            INNER JOIN patient_motion ON patient_motion.id = trial.patient_motion_id
            WHERE patient_motion.motion_id = ? AND gradient_set.sensor_id = ?
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
            INNER JOIN patient_motion ON patient_motion.id = trial.patient_motion_id
            WHERE patient_motion.motion_id = ? AND gradient_set.sensor_id = ?
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
            INNER JOIN patient_motion ON patient_motion.id = trial.patient_motion_id
            WHERE patient_motion.motion_id = ?
        """

        self._cursor.execute(query, (self.id,))
        position_sets = [PositionSet(*row) for row in self._cursor.fetchall()]
        return position_sets

    def get_grad_sets(self):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        query = f"""
            SELECT gradient_set.*
            FROM gradient_set
            INNER JOIN trial ON trial.id = gradient_set.trial_id
            INNER JOIN patient_motion ON patient_motion.id = trial.patient_motion_id
            WHERE patient_motion.motion_id = ?
        """

        self._cursor.execute(query, (self.id,))
        gradient_sets = [GradientSet(*row) for row in self._cursor.fetchall()]
        return gradient_sets

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

    def get_pos_set_matrices(self):
        pos_sets = self.get_pos_sets()
        pos_set_matrices = [pos_set.get_matrix("matrix") for pos_set in pos_sets]
        return pos_set_matrices



    # def get_box_plt(self, sensor):
    #     grad_sets = self.get_gradient_sets_for_sensor(sensor)
    #     name = sensor.name
    #     for gs in grad_sets:



    def __str__(self) -> str:
        return self.description
