import os
import sqlite3
from generator import Generator
import pdb

exp = {}
root_dir = "controls_alignedCoordinateSystem"
conn = sqlite3.connect('motion_analysis.db')
cursor = conn.cursor()

# Create the Patient, Motion, and PatientMotion tables
cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL UNIQUE
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS motion (
        id INTEGER PRIMARY KEY,
        description TEXT NOT NULL
    )
""")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS patient_motion (
        id INTEGER PRIMARY KEY,
        patient_id INTEGER,
        motion_id INTEGER,
        FOREIGN KEY (patient_id) REFERENCES patient (id),
        FOREIGN KEY (motion_id) REFERENCES motion (id)
    )
""")


for subdir, _, files in os.walk(root_dir):
    for file in files:
        file_path = os.path.join(subdir, file)
        if file.endswith('.npy'):
            v = Generator(file_path)
            exp[v.name] = v

conn.commit()
print("Done!")
