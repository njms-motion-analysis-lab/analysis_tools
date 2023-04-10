# test_models.py
import sqlite3
import unittest
import sqlite3
import os
from models.patient import Patient
from models.motion import Motion  # Replace 'your_module' with the module containing your classes
from models.base_model import BaseModel

class TestModels(unittest.TestCase):

    @classmethod
    def tearDownClass(cls):
        with sqlite3.connect('motion_analysis_test.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patient")
            cursor.execute("DELETE FROM motion")
            cursor.execute("DELETE FROM patient_motion")
            cursor.execute("DELETE FROM trial")
            cursor.execute("DELETE FROM sensor")
            cursor.execute("DELETE FROM position_set")

        os.remove('motion_analysis_test.db')

    def setUp(self) -> None:
        with sqlite3.connect('motion_analysis_test.db') as conn:
            self.cursor = conn.cursor()
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS motion (
                    id INTEGER PRIMARY KEY,
                    description TEXT NOT NULL
                )
            """)
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS patient_motion (
                    id INTEGER PRIMARY KEY,
                    patient_id INTEGER,
                    motion_id INTEGER,
                    UNIQUE (patient_id, motion_id),
                    FOREIGN KEY (patient_id) REFERENCES patient (id),
                    FOREIGN KEY (motion_id) REFERENCES motion (id)
                )
            """)
    
            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS trial (
                    id INTEGER PRIMARY KEY,
                    patient_motion_id INTEGER NOT NULL,
                    timestamp DATETIME,
                    FOREIGN KEY (patient_motion_id) REFERENCES patient_motion (id)
                )
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS sensor (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    axis TEXT,
                    part TEXT,
                    side TEXT,
                    placement TEXT,
                    kind TEXT,
                    UNIQUE(axis, part, side, placement)
                )
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS position_set (
                    id INTEGER PRIMARY KEY,
                    sensor_id INTEGER,
                    trial_id INTEGER,
                    matrix TEXT,
                    FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                    FOREIGN KEY (trial_id) REFERENCES trial (id)
                )
            """)

            self.cursor.execute("""
                CREATE TABLE IF NOT EXISTS gradient_set (
                    id INTEGER PRIMARY KEY,
                    sensor_id INTEGER,
                    trial_id INTEGER,
                    matrix TEXT,
                    FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                    FOREIGN KEY (trial_id) REFERENCES trial (id)
                )
            """)

        BaseModel.set_class_connection(test_mode=True)
        
    def tearDown(self) -> None:
        with sqlite3.connect('motion_analysis_test.db') as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM patient")
            cursor.execute("DELETE FROM motion")
            cursor.execute("DELETE FROM patient_motion")
            cursor.execute("DELETE FROM trial")
            cursor.execute("DELETE FROM sensor")
            cursor.execute("DELETE FROM position_set")

    def test_create_patient(self):
        patient = Patient(name="John Doe")
        created = patient.create()
        self.assertTrue(created, "Patient record not created")

    def test_create_motion(self):
        motion = Motion(description="Jumping")
        created = motion.create()
        self.assertTrue(created, "Motion record not created")

    def test_patient_motion_relationship(self):
        BaseModel.delete_all_and_children()
        patient = Patient(name="John Doe")
        patient.create()
        motion = Motion(description="Jumping")
        motion.create()

        patient.add_motion(motion)  # Add the motion to the patient

        patient_motions = patient.get_motions()
        self.assertEqual(len(patient_motions), 1, "Motion not added to patient")

        print(patient_motions)
        self.assertEqual(patient_motions[0].id, motion.id, "Incorrect motion added to patient")

        BaseModel.delete_all_and_children()
        patient = Patient(name="Bob")
        patient.create()
        motion = Motion(description="Sleeping")
        motion.create()


        motion.add_patient(patient)
        motion_patients = motion.get_patients()
        for mp in motion_patients:
            print("Motion patient:", mp.id, mp.name)
            
        self.assertEqual(len(motion_patients), 1, "Patient not added to motion")
        self.assertEqual(motion_patients[0].id, patient.id, "Incorrect patient added to motion")

        patient.remove_motion(motion)

        patient_motions = patient.get_motions()
        self.assertEqual(len(patient_motions), 0, "Motion not removed from patient")

if __name__ == "__main__":
    unittest.main()