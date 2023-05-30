import threading
import sqlite3

class Database:
    _instance = None
    lock = threading.Lock()

    def __init__(self):
        if self._instance is not None:
            raise Exception("This is a singleton class.")
        self.connection = sqlite3.connect('motion_analysis.db')
        self.cursor = self.connection.cursor()

    @classmethod
    def get_instance(cls):
        with cls.lock:
            if cls._instance is None:
                cls._instance = Database()
        return cls._instance

    @classmethod
    def view_current_tables(cls):
        with cls.lock:
            cls.get_instance().cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            print("Current Tables in the database are: ")
            tables = cls.get_instance().cursor.fetchall()
            for table in tables:
                print(table[0])


    @classmethod
    def is_database_locked(cls):
        try:
            cursor = cls.get_instance().cursor
            cursor.execute("""CREATE TABLE IF NOT EXISTS temp_table (
                                id INTEGER PRIMARY KEY
                            );""")
            cls.get_instance().connection.commit()
            cursor.execute("DROP TABLE temp_table;")
            cls.get_instance().cursor.commit()
            return False
        except sqlite3.OperationalError as e:
            if str(e) == "database is locked":
                return True
            else:
                raise e