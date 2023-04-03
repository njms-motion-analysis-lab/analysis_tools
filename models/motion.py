import sqlite3
from models.base_model import BaseModel


# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.

class Motion(BaseModel):
    table_name = "motion"

    def __init__(self, id=None, description=None):
        self.description = description
        self.id = id
        self.patient_id = None
    
    def create(self, **kwargs):
        if self.description is not None:
            row_id = super().create(description=self.description)
        else:
            row_id = super().create(**kwargs)
        self.id = row_id
        return True

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
        #
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

    
    def __str__(self) -> str:
        return self.description