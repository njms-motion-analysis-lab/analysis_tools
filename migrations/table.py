from models.base_model import BaseModel
from database import Database
import psycopg2

class Table:
    conn = Database.get_instance().connection
    cursor = conn.cursor()
    @classmethod
    def create_tables(cls):
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient (
                id SERIAL PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

    # change to task
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS task (
                id SERIAL PRIMARY KEY,
                description TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_task (
                id SERIAL PRIMARY KEY,
                patient_id INTEGER NOT NULL,
                task_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE (patient_id, task_id),
                FOREIGN KEY (patient_id) REFERENCES patient (id),
                FOREIGN KEY (task_id) REFERENCES task (id)
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS trial (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                patient_task_id INTEGER NOT NULL,
                trial_num INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE (patient_task_id, trial_num, name),
                FOREIGN KEY (patient_task_id) REFERENCES patient_task (id)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                axis TEXT,
                part TEXT,
                side TEXT,
                placement TEXT,
                kind TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                UNIQUE(axis, part, side, placement)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS position_set (
                id SERIAL PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix BYTEA,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_position_set (
                id SERIAL PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix BYTEA,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS gradient_set (
                id SERIAL PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix BYTEA,
                aggregated_stats BYTEA,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_gradient_set (
                id SERIAL PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix BYTEA,
                aggregated_stats BYTEA,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sub_gradient (
                id SERIAL PRIMARY KEY,
                name TEXT,
                valid BOOLEAN,
                matrix BYTEA,
                gradient_set_id INTEGER NOT NULL,
                gradient_set_ord INTEGER,
                start_time INTEGER,
                stop_time INTEGER,
                mean REAL,
                median REAL,
                stdev REAL,
                normalized BYTEA,
                submovement_stats BYTEA,
                submovement_stats_nonnorm BYTEA,
                submovement_stats_position BYTEA,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                UNIQUE (gradient_set_id, gradient_set_ord),
                FOREIGN KEY (gradient_set_id) REFERENCES gradient_set (id)
            );
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_sub_gradient (
                id SERIAL PRIMARY KEY,
                name TEXT,
                valid BOOLEAN,
                matrix BYTEA,
                dynamic_gradient_set_id INTEGER NOT NULL,
                dynamic_gradient_set_ord INTEGER,
                start_time INTEGER,
                stop_time INTEGER,
                normalized BYTEA,
                submovement_stats TEXT,
                created_at TIMESTAMP,
                updated_at TIMESTAMP,
                FOREIGN KEY (dynamic_gradient_set_id) REFERENCES dynamic_gradient_set (id)
            );
        """)

        # Classifier table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS classifier (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        # Params and Hyperparams table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS params (
                id INTEGER PRIMARY KEY,
                classifier_id INTEGER NOT NULL,
                params TEXT NOT NULL,
                hyperparams TEXT NOT NULL,
                FOREIGN KEY (classifier_id) REFERENCES classifier (id)
            )
        """)

        # Scores table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY,
                params_id INTEGER NOT NULL,
                score REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (params_id) REFERENCES params (id)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY,
                params_id INTEGER NOT NULL,
                params_id INTEGER NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (params_id) REFERENCES params (id)
                FOREIGN KEY (scoress_id) REFERENCES scores (id)
            )
        """)
        cls.conn.commit()
    

    @classmethod
    def column_exists(cls, table_name, column_name):
        cls.cursor.execute(f"""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name='{table_name}' and column_name='{column_name}';
        """)
        return bool(cls.cursor.fetchone())

    @classmethod
    def add_column_if_not_exists(cls, table_name, column_name, column_type):
        print(table_name, column_name)
        if not cls.column_exists(table_name, column_name):
            print(table_name, column_name)
            cls.cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")
            cls.conn.commit()

    @classmethod
    def update_tables(cls):
        # List of columns to be added, and their types
        new_columns = {
            "mean": "BYTEA",
            "median": "BYTEA",
            "stdev": "BYTEA",
            "submovement_stats_nonnorm": "BYTEA",
            "submovement_stats_position": "BYTEA",
        }

        # Add new columns to dynamic_sub_gradient table if they do not exist
        for column, column_type in new_columns.items():
            cls.add_column_if_not_exists('dynamic_sub_gradient', column, column_type)
        cls.add_column_if_not_exists('trial', "is_dominant", "BOOLEAN")
        cls.add_column_if_not_exists('task', "is_dominant", "BOOLEAN")
        cls.add_column_if_not_exists('patient', "dominant_side", "TEXT")
        cls.add_column_if_not_exists('gradient_set', "features_extracted", "TIMESTAMP")
        print("Done!!")
        

        cls.cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_dynamic_sub_gradient_set_id_ord 
            ON dynamic_sub_gradient (dynamic_gradient_set_id, dynamic_gradient_set_ord)
        """)

    @classmethod
    def drop_all_tables(cls):
        try:
            cls.cursor.execute("SELECT pg_terminate_backend(pg_stat_activity.pid) FROM pg_stat_activity WHERE pg_stat_activity.datname = current_database() AND pid <> pg_backend_pid();")
            cls.conn.commit()
            print("Open transactions terminated.")
        except psycopg2.Error as e:
            print(f"Error terminating open transactions: {e}")
            cls.conn.rollback()

        tables = ['sub_gradient', 'dynamic_sub_gradient', 'gradient_set', 'dynamic_gradient_set', 'position_set',
                  'dynamic_position_set', 'patient_task', 'trial', 'sensor', 'task', 'patient', 'motion', 'patient_motion']

        for table in tables:
            try:
                cls.cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                cls.conn.commit()
                print(f"Table {table} dropped.")
            except psycopg2.Error as e:
                print(f"Error dropping table: {e}")
                cls.conn.rollback()

        print("All tables dropped.")

    @classmethod
    def remove_deadlock(cls):
        cls.cursor.execute("SELECT pg_terminate_backend(pid) FROM pg_locks WHERE NOT granted;")
        cls.conn.commit()
        print("Deadlock removed.")

    @classmethod
    def clear_tables(cls):
        # delete all records from patient_task that have patient_id in patient table
        cls.cursor.execute("DELETE FROM trial")
        cls.cursor.execute("DELETE FROM patient_task WHERE patient_id IN (SELECT id FROM patient)")
        # now, you can delete records from patient table
        cls.cursor.execute("DELETE FROM patient")
        cls.cursor.execute("DELETE FROM task")
        cls.cursor.execute("DELETE FROM patient_task")
        
        cls.cursor.execute("DELETE FROM sensor")
        cls.cursor.execute("DELETE FROM position_set")
        cls.cursor.execute("DELETE FROM dynamic_position_set")
        cls.cursor.execute("DELETE FROM sub_gradient")
        cls.cursor.execute("DELETE FROM dynamic_sub_gradient")
        cls.cursor.execute("DELETE FROM gradient_set")
        cls.cursor.execute("DELETE FROM dynamic_gradient_set")
        cls.conn.commit()

    