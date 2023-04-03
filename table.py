import sqlite3
from models.base_model import BaseModel

class Table:
    conn = sqlite3.connect('motion_analysis.db')
    cursor = conn.cursor()
    @classmethod
    def create_tables(cls):

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
                UNIQUE (patient_id, motion_id),
                FOREIGN KEY (patient_id) REFERENCES patient (id),
                FOREIGN KEY (motion_id) REFERENCES motion (id)
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trial (
                id INTEGER PRIMARY KEY,
                patient_motion_id INTEGER NOT NULL,
                timestamp DATETIME,
                FOREIGN KEY (patient_motion_id) REFERENCES patient_motion (id)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                axis TEXT,
                part TEXT,
                side TEXT,
                iteration TEXT,
                kind TEXT,
                UNIQUE(axis, part, side, iteration)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_set (
                id INTEGER PRIMARY KEY,
                sensor_id INTEGER,
                trial_id INTEGER,
                matrix TEXT,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS gradient_set (
                id INTEGER PRIMARY KEY,
                sensor_id INTEGER,
                trial_id INTEGER,
                matrix TEXT,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id)
            )
        """)

        
        BaseModel.set_connection(conn=cls.conn, cursor=cls.cursor)
    
    @classmethod
    def clear_tables(cls):
        cls.cursor.execute("DELETE FROM patient")
        cls.cursor.execute("DELETE FROM motion")
        cls.cursor.execute("DELETE FROM patient_motion")
        cls.cursor.execute("DELETE FROM trial")
        cls.cursor.execute("DELETE FROM sensor")
        cls.cursor.execute("DELETE FROM position_set")
        cls.conn.commit()

    
