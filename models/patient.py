import sqlite3
from models.base_model import BaseModel
from models.motion import Motion
from models.trial import Trial

class Patient(BaseModel):
    table_name = "patient"
    # A subclass of the `BaseModel` class, representing a patient in the database.
    # The `table_name` class attribute specifies the name of the database table where patient data is stored.
    
    def __init__(self, id=None, name=None):
        self.name = name
        self.id = id

    def create(self, **kwargs):
        if self.name is not None:
            row_id = super().create(name=self.name)
        else:
            row_id = super().create(**kwargs)

        self.id = row_id
        return True

    def add_motion(self, motion):
        # Check if the relationship already exists
        self._cursor.execute("SELECT * FROM patient_motion WHERE patient_id=? AND motion_id=?", (self.id, motion.id))
        existing_relation = self._cursor.fetchone()

        if not existing_relation:
            # Add a motion to the list of motions associated with the patient.
            self._cursor.execute("INSERT INTO patient_motion (patient_id, motion_id) VALUES (?, ?)", (self.id, motion.id))
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
    
    def __str__(self) -> str:
        # Return the string representation of the patient (the patient's name).
        #
        # Input:
        # - None.
        #
        # Output:
        # - A string representing the patient (the patient's name).
        return self.name