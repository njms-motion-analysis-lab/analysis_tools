import sqlite3
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel

class Table:
    conn = sqlite3.connect('motion_analysis.db')
    cursor = conn.cursor()
    @classmethod
    def create_tables(cls):
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

    # change to task
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS task (
                id INTEGER PRIMARY KEY,
                description TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_task (
                id INTEGER PRIMARY KEY,
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
                id INTEGER PRIMARY KEY,
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
                id INTEGER PRIMARY KEY,
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
                id INTEGER PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_position_set (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS gradient_set (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix TEXT,
                aggregated_stats TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_gradient_set (
                id INTEGER PRIMARY KEY,
                name TEXT,
                sensor_id INTEGER NOT NULL,
                trial_id INTEGER NOT NULL,
                matrix TEXT,
                aggregated_stats TEXT,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (sensor_id) REFERENCES sensor (id),
                FOREIGN KEY (trial_id) REFERENCES trial (id),
                UNIQUE(trial_id, sensor_id, name)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS sub_gradient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                valid BOOLEAN,
                matrix TEXT,
                gradient_set_id INTEGER NOT NULL,
                gradient_set_ord INTEGER,
                start_time INTEGER,
                stop_time INTEGER,
                mean REAL,
                median REAL,
                stdev REAL,
                normalized TEXT,
                submovement_stats TEXT,
                submovement_stats_nonnorm TEXT,
                submovement_stats_position TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                UNIQUE (gradient_set_id, gradient_set_ord),
                FOREIGN KEY (gradient_set_id) REFERENCES gradient_set (id)
            );
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS dynamic_sub_gradient (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                valid BOOLEAN,
                matrix TEXT,
                dynamic_gradient_set_id INTEGER NOT NULL,
                dynamic_gradient_set_ord INTEGER,
                start_time INTEGER,
                stop_time INTEGER,
                normalized TEXT,
                submovement_stats TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                FOREIGN KEY (dynamic_gradient_set_id) REFERENCES dynamic_gradient_set (id)
            );
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictor (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id INTEGER,
                sensor_id INTEGER,
                non_norm INTEGER,
                abs_val INTEGER,
                accuracies TEXT,
                matrix TEXT,
                FOREIGN KEY (task_id) REFERENCES Tasks(id),
                FOREIGN KEY (sensor_id) REFERENCES Sensors(id)
            );
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS multi_predictor (
                id INTEGER PRIMARY KEY,
                task_id INTEGER,
                codes_score TEXT,
                model TEXT,
                items TEXT,
                FOREIGN KEY(task_id) REFERENCES Task(id)
            );
        """)


        # Classifier table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS classifier (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
        """)

        # Params and Hyperparams table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS params (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                classifier_id INTEGER NOT NULL,
                params TEXT NOT NULL,  -- Serialized parameters
                hyperparams TEXT NOT NULL,  -- Serialized hyperparameters
                features TEXT NOT NULL,  -- Serialized feature list
                FOREIGN KEY (classifier_id) REFERENCES classifier (id)
            )
        """)

        # Scores table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                params_id INTEGER NOT NULL,
                average_score REAL NOT NULL,
                classifier_accuracies TEXT NOT NULL,  -- Serialized accuracies
                rf_auc_roc REAL NOT NULL,
                rf_accuracy REAL NOT NULL,
                rf_confusion_matrix TEXT NOT NULL,  -- Serialized confusion matrix
                rf_f1_score REAL NOT NULL,
                rf_log_loss REAL NOT NULL,
                rf_precision REAL NOT NULL,
                rf_recall REAL NOT NULL,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (params_id) REFERENCES params (id)
            )
        """)

        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                params_id INTEGER NOT NULL,
                scores_id INTEGER NOT NULL,  -- Corrected the column name
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL,
                FOREIGN KEY (params_id) REFERENCES params (id)
                FOREIGN KEY (scores_id) REFERENCES scores (id)  -- Corrected the reference
            )
        """)

                # Session-Params join table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_params (
                session_id INTEGER NOT NULL,
                params_id INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES session (id),
                FOREIGN KEY (params_id) REFERENCES params (id),
                UNIQUE (session_id, params_id)
            )
        """)

        # Session-Scores join table
        cls.cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_scores (
                session_id INTEGER NOT NULL,
                scores_id INTEGER NOT NULL,
                FOREIGN KEY (session_id) REFERENCES session (id),
                FOREIGN KEY (scores_id) REFERENCES scores (id),
                UNIQUE (session_id, scores_id)
            )
        """)

        cls.conn.commit()
        
        LegacyBaseModel.set_class_connection()

    @classmethod
    def column_exists(cls, table_name, column_name):
        cls.cursor.execute(f"PRAGMA table_info({table_name})")
        columns = [column[1] for column in cls.cursor.fetchall()]
        return column_name in columns

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
            "mean": "REAL",
            "median": "REAL",
            "stdev": "REAL",
            "submovement_stats_nonnorm": "TEXT",
            "submovement_stats_position": "TEXT",
        }

        # Add new columns to dynamic_sub_gradient table if they do not exist
        for column, column_type in new_columns.items():
            cls.add_column_if_not_exists('dynamic_sub_gradient', column, column_type)
        cls.add_column_if_not_exists('trial', "is_dominant", "BOOLEAN")
        cls.add_column_if_not_exists('task', "is_dominant", "BOOLEAN")
        cls.add_column_if_not_exists('patient', "dominant_side", "TEXT")
        cls.add_column_if_not_exists('predictor', "created_at", "TEXT")
        cls.add_column_if_not_exists('predictor', "updated_at", "DATETIME")
        cls.add_column_if_not_exists('predictor', "multi_predictor_id", "INTEGER")
        cls.add_column_if_not_exists('predictor', "aggregated_stats", "TEXT")
        cls.add_column_if_not_exists('predictor', "aggregated_stats_non_normed", "TEXT")
        cls.add_column_if_not_exists('multi_predictor', "created_at", "TEXT")
        cls.add_column_if_not_exists('multi_predictor', "updated_at", "DATETIME")
        cls.add_column_if_not_exists('predictor', "multi_predictor_id", "INTEGER")
        print("Done!!")
        

        cls.cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_dynamic_sub_gradient_set_id_ord 
            ON dynamic_sub_gradient (dynamic_gradient_set_id, dynamic_gradient_set_ord)
        """)


    @classmethod
    def drop_all_tables(cls):
        tables = ['trial', 'patient_task', 'sensor', 'task', 'patient', 'position_set','dynamic_position_set', 'gradient_set', 'dynamic_gradient_set', 'dynamic_sub_gradient', 'motion', 'patient_motion', "sub_gradient"]
        for table in tables:
            try:
                BaseModel._cursor.execute(f"DROP TABLE IF EXISTS {table}")
                BaseModel._conn.commit()
            except sqlite3.OperationalError as e:
                print(f"Error dropping table: {e}")
                BaseModel._conn.rollback()


    @classmethod
    def clear_tables(cls):
        cls.cursor.execute("DELETE FROM patient")
        cls.cursor.execute("DELETE FROM task")
        cls.cursor.execute("DELETE FROM patient_task")
        cls.cursor.execute("DELETE FROM trial")
        cls.cursor.execute("DELETE FROM sensor")
        cls.cursor.execute("DELETE FROM position_set")
        cls.cursor.execute("DELETE FROM dynamic_position_set")
        cls.cursor.execute("DELETE FROM sub_gradient")
        cls.cursor.execute("DELETE FROM dynamic_sub_gradient")
        cls.cursor.execute("DELETE FROM gradient_set")
        cls.cursor.execute("DELETE FROM dynamic_gradient_set")
        cls.conn.commit()

    