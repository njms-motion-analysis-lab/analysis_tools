import sqlite3
from models.base_model import BaseModel
from models.motion import Motion
from models.patient_motion import PatientMotion
from models.trial import Trial
import pdb
from datetime import datetime
class Patient(BaseModel):
    table_name = "patient"
    _conn = BaseModel._conn
    _cursor = BaseModel._cursor
    # A subclass of the `BaseModel` class, representing a patient in the database.
    # The `table_name` class attribute specifies the name of the database table where patient data is stored.
    
    def __init__(self, id=None, name=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.created_at = created_at
        self.updated_at = updated_at

    def add_motion(self, motion):
        # Check if the relationship already exists
        self._cursor.execute("SELECT * FROM patient_motion WHERE patient_id=? AND motion_id=?", (self.id, motion.id))
        existing_relation = self._cursor.fetchone()

        if not existing_relation:
            curr = datetime.now()
            # Add a motion to the list of motions associated with the patient.
            self._cursor.execute("INSERT INTO patient_motion (patient_id, motion_id, created_at, updated_at) VALUES (?, ?, ?, ?)", (self.id, motion.id, curr, curr))
            self._conn.commit()
        else:
            print("The relationship between this patient and motion already exists.")

    def remove_motion(self, motion):
        # Remove a motion from the list of motions associated with the patient.
        #
        # Input:
        # - `motion`: the `Motion` object to remove from the patient's list of motions.
        #
        # Output:
        # - None.
        self._cursor.execute("DELETE FROM patient_motion WHERE patient_id=? AND motion_id=?", (self.id, motion.id))
        self._conn.commit()

    def get_patient_motion_by_motion(self, motion):
        """
        Return the PatientMotion associated with the given Motion for this Patient instance.

        :param motion: The Motion instance to find the associated PatientMotion.
        :return: The associated PatientMotion instance if found, None otherwise.
        """

        return PatientMotion.where(motion=motion, patient=self)[0]

    


    def get_motions(self):
        self._cursor.execute("""
            SELECT motion.* FROM motion
            JOIN patient_motion ON motion.id = patient_motion.motion_id
            WHERE patient_motion.patient_id = ?
        """, (self.id,))

        return [Motion.get(row[0]) for row in self._cursor.fetchall()]

    def get_trials(self):
        self._cursor.execute("""
            SELECT trial.* FROM trial
            JOIN patient_motion ON trial.patient_motion_id = patient_motion.id
            WHERE patient_motion.patient_id = ?
        """, (self.id,))

        return [Trial(*row) for row in self._cursor.fetchall()]
    
    @classmethod
    def delete_all(cls):
        # Delete records from the join table
        cls._cursor.execute(f"DELETE FROM patient_motions WHERE patient_id IN (SELECT id FROM {cls.table_name})")

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