import threading
import psycopg2

class Database:
    _instance = None
    lock = threading.Lock()

    def __init__(self):
        if self._instance is not None:
            raise Exception("This is a singleton class.")
        self.connection = psycopg2.connect(
            dbname="motion_analysis",
            user="postgres",
            password="password",
            host="localhost",
            port="5432"
        )
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
            cls.get_instance().cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            print("Current Tables in the database are: ")
            tables = cls.get_instance().cursor.fetchall()
            for table in tables:
                print(table[0])

    @classmethod
    def is_database_locked(cls):
        # This function is not applicable to PostgreSQL, so it could be removed
        pass
