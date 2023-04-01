import sqlite3
from models.base_model import BaseModel

# Connect to the SQLite database named 'motion_analysis.db' and create a cursor object for executing SQL commands.
conn = sqlite3.connect('motion_analysis.db')
cursor = conn.cursor()

class Motion(BaseModel):
    # A subclass of the `BaseModel` class, representing a motion in the database.
    # The `table_name` class attribute specifies the name of the database table where motion data is stored.
    table_name = "motion"

    def __init__(self, id, description, patient_id=None):
        # Initialize a new `Motion` object with the specified `id`, `description`, and `patient_id`.
        #
        # Input:
        # - `id`: the unique identifier for the motion.
        # - `description`: a description of the motion.
        # - `patient_id`: the unique identifier of the patient associated with the motion (optional).
        #
        # Output:
        # - None.
        self.id = id
        self.description = description
        self.patient_id = patient_id

    def get_patients(self):
        # Retrieve a list of patients associated with the motion.
        #
        # Input:
        # - None.
        #
        # Output:
        # - A list of `Patient` objects associated with the motion.
        from models.patient import Patient
        cursor.execute("""
            SELECT patient.* FROM patient
            JOIN patient_motion ON patient.id = patient_motion.patient_id
            WHERE patient_motion.motion_id = ?
        """, (self.id,))
        return [Patient(*row) for row in cursor.fetchall()]
    
    def __str__(self) -> str:
        # Return the string representation of the motion (the motion's description).
        #
        # Input:
        # - None.
        #
        # Output:
        # - A string representing the motion (the motion's description).
        return self.description