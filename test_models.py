# test_models.py
import sqlite3
import unittest
import sqlite3
import os
from models.patient import Patient
from models.motion import Motion

class TestModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect('motion_analysis.db')
        cls.cursor = cls.conn.cursor()
        # Create the Patient, Motion, and PatientMotion tables
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS motion (
                id INTEGER PRIMARY KEY,
                description TEXT NOT NULL
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_motion (
                id INTEGER PRIMARY KEY,
                patient_id INTEGER,
                motion_id INTEGER,
                FOREIGN KEY (patient_id) REFERENCES patient (id),
                FOREIGN KEY (motion_id) REFERENCES motion (id)
            )
        """)

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()
        os.remove('motion_analysis.db')

    def setUp(self):
        self.cursor.execute("DELETE FROM patient")
        self.cursor.execute("DELETE FROM motion")
        self.cursor.execute("DELETE FROM patient_motion")
        self.conn.commit()

    def test_create_patient(self):
        patient = Patient(None, "John Doe")
        created = patient.create(name="John Doe")
        self.assertTrue(created, "Patient record not created")
        self.assertIsNotNone(patient.id, "Patient ID not assigned")

    def test_create_motion(self):
        motion = Motion(None, "Jumping")
        created = motion.create(description="Jumping")
        self.assertTrue(created, "Motion record not created")
        self.assertIsNotNone(motion.id, "Motion ID not assigned")

    def test_patient_motion_relationship(self):
        patient = Patient(None, "John Doe")
        patient.create(name="John Doe")
        motion = Motion(None, "Jumping")
        motion.create(description="Jumping")

        patient.add_motion(motion)

        patient_motions = patient.get_motions()
        self.assertEqual(len(patient_motions), 1, "Motion not added to patient")
        self.assertEqual(patient_motions[0].id, motion.id, "Incorrect motion added to patient")

        motion_patients = motion.get_patients()
        self.assertEqual(len(motion_patients), 1, "Patient not added to motion")
        self.assertEqual(motion_patients[0].id, patient.id, "Incorrect patient added to motion")

        patient.remove_motion(motion)

        patient_motions = patient.get_motions()
        self.assertEqual(len(patient_motions), 0, "Motion not removed from patient")

if __name__ == "__main__":
    unittest.main()