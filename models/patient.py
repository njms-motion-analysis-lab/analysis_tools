import sqlite3
from models.base_model import BaseModel
from models.motion import Motion

# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.
conn = sqlite3.connect('motion_analysis.db')
cursor = conn.cursor()

class Patient(BaseModel):
    # A subclass of the `BaseModel` class, representing a patient in the database.
    # The `table_name` class attribute specifies the name of the database table where patient data is stored.
    table_name = "patient"

    def __init__(self, id, name):
        # Initialize a new `Patient` object with the specified `id` and `name`.
        #
        # Input:
        # - `id`: the unique identifier for the patient.
        # - `name`: the name of the patient.
        #
        # Output:
        # - None.
        self.id = id
        self.name = name

    def add_motion(self, motion):
        # Add a motion to the list of motions associated with the patient.
        #
        # Input:
        # - `motion`: the `Motion` object to add to the patient's list of motions.
        #
        # Output:
        # - None.
        cursor.execute("INSERT INTO patient_motion (patient_id, motion_id) VALUES (?, ?)", (self.id, motion.id))
        conn.commit()

    def remove_motion(self, motion):
        # Remove a motion from the list of motions associated with the patient.
        #
        # Input:
        # - `motion`: the `Motion` object to remove from the patient's list of motions.
        #
        # Output:
        # - None.
        cursor.execute("DELETE FROM patient_motion WHERE patient_id=? AND motion_id=?", (self.id, motion.id))
        conn.commit()

    def get_motions(self):
        # Retrieve a list of motions associated with the patient.
        #
        # Input:
        # - None.
        #
        # Output:
        # - A list of `Motion` objects associated with the patient.
        cursor.execute("""
            SELECT motion.* FROM motion
            JOIN patient_motion ON motion.id = patient_motion.motion_id
            WHERE patient_motion.patient_id = ?
        """, (self.id,))
        return [Motion(*row) for row in cursor.fetchall()]
    
    def __str__(self) -> str:
        # Return the string representation of the patient (the patient's name).
        #
        # Input:
        # - None.
        #
        # Output:
        # - A string representing the patient (the patient's name).
        return self.name