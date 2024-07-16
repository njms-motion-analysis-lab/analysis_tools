
from importlib import import_module
import sqlite3
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import pdb
from datetime import datetime

from models.legacy_position_set import PositionSet
from models.legacy_sensor import Sensor
class Patient(LegacyBaseModel):
    table_name = "patient"
    _conn = LegacyBaseModel._conn
    _cursor = LegacyBaseModel._cursor
    # A subclass of the `LegacyBaseModel` class, representing a patient in the database.
    # The `table_name` class attribute specifies the name of the database table where patient data is stored.
    
    def __init__(self, id=None, name=None, created_at=None, updated_at=None, dominant_side="R", cohort_id=None):
        super().__init__()
        self.id = id
        self.name = name
        self.created_at = created_at
        self.updated_at = updated_at
        self.dominant_side = dominant_side
        self.cohort_id = cohort_id

    def add_task(self, task):
        from importlib import import_module
        Task = import_module("models.legacy_task").Task
        # Check if the relationship already exists
        self._cursor.execute("SELECT * FROM patient_task WHERE patient_id=? AND cohort_id=? AND task_id=?", (self.id, self.cohort_id, task.id))
        existing_relation = self._cursor.fetchone()

        if not existing_relation:
            curr = datetime.now()
            # Add a task to the list of tasks associated with the patient.
            self._cursor.execute("INSERT INTO patient_task (patient_id, task_id, cohort_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?)", (self.id, task.id, self.cohort_id, curr, curr))
            self._conn.commit()
        else:
            print("The relationship between this patient and task already exists.")

    def remove_task(self, task):
        # Remove a task from the list of tasks associated with the patient.
        #
        # Input:
        # - `task`: the `Task` object to remove from the patient's list of tasks.
        #
        # Output:
        # - None.
        self._cursor.execute("DELETE FROM patient_task WHERE patient_id=? AND task_id=?", (self.id, task.id))
        self._conn.commit()

    def patient_task_by_task(self, task):
        from importlib import import_module
        PatientTask = import_module("models.legacy_patient_task").PatientTask
        """
        Return the PatientTask associated with the given Task for this Patient instance.

        :param task: The Task instance to find the associated PatientTask.
        :return: The associated PatientTask instance if found, None otherwise.
        """
        pt = PatientTask.where(task_id=task.id, patient_id=self.id)
        if len(pt) == 0:
            return None
        return pt[0]

    
    def tasks(self):
        from importlib import import_module
        Task = import_module("models.legacy_task").Task
        self._cursor.execute("""
            SELECT task.* FROM task
            JOIN patient_task ON task.id = patient_task.task_id
            WHERE patient_task.patient_id = ?
        """, (self.id,))

        return [Task.get(row[0]) for row in self._cursor.fetchall()]

    def trials(self, *attributes):
        from importlib import import_module
        Trial = import_module("models.legacy_trial").Trial
        self._cursor.execute("""
            SELECT trial.* FROM trial
            JOIN patient_task ON trial.patient_task_id = patient_task.id
            WHERE patient_task.patient_id = ?
        """, (self.id,))
        
        trials_list = [Trial(*row) for row in self._cursor.fetchall()]
        
        if attributes:
            return [
                tuple(getattr(trial, attribute) for attribute in attributes)
                for trial in trials_list
            ]
        return trials_list
    
    @classmethod
    def delete_all(cls):
        # Delete records from the join table
        cls._cursor.execute(f"DELETE FROM patient_tasks WHERE patient_id IN (SELECT id FROM {cls.table_name})")

        # Delete records from the current class table
        cls._cursor.execute(f"DELETE FROM {cls.table_name}")
    

    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()
    
    def __str__(self) -> str:
        # Return the string representation of the patient (the patient's name).
        #
        # Input:
        # - None.
        #
        # Output:
        # - A string representing the patient (the patient's name).
        return self.name