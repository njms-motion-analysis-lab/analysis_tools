from datetime import datetime
import pickle
import sqlite3
import pandas as pd


class BaseModel:
    table_name = ""
    _conn = sqlite3.connect('motion_analysis.db')
    _cursor = _conn.cursor()

    def __init__(self, id=None, created_at=None, updated_at=None, **kwargs):
        self.id = id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
    
    @classmethod
    def set_class_connection(cls, test_mode=False, conn=None, cursor=None):
        if test_mode is True:
            cls._conn = sqlite3.connect('motion_analysis_test.db')
            cls._cursor = cls._conn.cursor()
        if (conn is not None) and (cursor is not None):
            cls._conn = conn
            cls._cursor = cursor
        print("set_class_connection _cursor:", cls._cursor)

    def set_connection(self, test_mode=False, conn=None, cursor=None):
        if test_mode is True:
            self.__class__._conn = sqlite3.connect('motion_analysis_test.db')
            self.__class__._cursor = self.__class__._conn.cursor()
        if (conn is not None) and (cursor is not None):
            self.__class__._conn = conn
            self.__class__._cursor = cursor
        print("set_connection _cursor:", self.__class__._cursor)

    def create(self, **kwargs):
        keys = ', '.join(['id', 'created_at', 'updated_at'] + list(kwargs.keys()))
        values = ', '.join(['?', '?', '?'] + ['?'] * len(kwargs))
        
        now = datetime.now()
        self.created_at = now
        self.updated_at = now
        with self.__class__._conn:
            try:
                self.__class__._cursor.execute(f"INSERT INTO {self.table_name} ({keys}) VALUES ({values})", (self.id, self.created_at, self.updated_at) + tuple(kwargs.values()))
                self.id = self.__class__._cursor.lastrowid
                self.__class__._conn.commit()
                return self.id
            except sqlite3.IntegrityError as e:
                print(f"Error creating record: {e}")

    def update(self, **kwargs):
        updates = ", ".join(f"{key}=?" for key in kwargs)
        updates += ", updated_at=?"  # Automatically update the updated_at timestamp

        now = datetime.now()
        self.updated_at = now
        try:
            self.__class__._cursor.execute(f"UPDATE {self.table_name} SET {updates} WHERE id=?", tuple(kwargs.values()) + (self.updated_at, self.id,))
            self.__class__._conn.commit()
            
            # Update the in-memory instance attributes
            for key, value in kwargs.items():
                setattr(self, key, value)
        except sqlite3.IntegrityError as e:
            print(f"Error updating record: {e}")
            return False
        return True


    def delete(self):
        self.__class__._cursor.execute(f"DELETE FROM {self.table_name} WHERE id=?", (self.id,))
        self.__class__._conn.commit()

    def get_matrix(self, column_name):
        # Fetch the binary data for the specified column from the database
        self.__class__._cursor.execute(f"SELECT {column_name} FROM {self.table_name} WHERE id=?", (self.id,))
        row = self.__class__._cursor.fetchone()

        if row:
            # Deserialize the binary data and return it as a numpy array
            return pickle.loads(row[0])
        else:
            return None

    def attrs(self):
        attributes = vars(self)
        print("Attributes:")
        for attribute, value in attributes.items():
            print(f"{attribute}: {value}")

    @classmethod
    def get(cls, id):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE id=?", (id,))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return "nope"

    @classmethod
    def find_by(cls, column_name, value):
        # Find the record with the given column name and value
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {column_name}=?", (value,))
        row = cls._cursor.fetchone()

        if row:
            return cls(*row)
        else:
            return None

    @classmethod
    def find_or_create(cls, **kwargs):
        keys = kwargs.keys()
        if 'matrix' in kwargs:
            old_matrix = kwargs["matrix"]
            kwargs["matrix"] = memoryview(pickle.dumps(kwargs["matrix"]))
        values = tuple(kwargs.values())
        # Find the record with the given attribute(s)
        conditions = " AND ".join(f"{key}=?" for key in keys)
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {conditions}", values)
        row = cls._cursor.fetchone()

        if row:
            return cls(*row)
        else:
            if 'matrix' in kwargs:
                kwargs["matrix"] = memoryview(pickle.dumps(old_matrix))
                # Serialize the matrix
                # Create the record if not found
                
            cls_instance = cls()
            
            for key, value in kwargs.items():
                setattr(cls_instance, key, value)
            cls_instance.id = cls_instance.create(**kwargs)
            return cls_instance
    
    @classmethod
    def all(cls):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name}")
        return [cls(*row) for row in cls._cursor.fetchall()]

    @classmethod
    def last(cls, n):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} ORDER BY updated_at DESC LIMIT ?", (n,))
        rows = cls._cursor.fetchall()
        return [cls(*row) for row in rows]

    @classmethod
    def delete_all(cls):
        cls._cursor.execute(f"DELETE FROM {cls.table_name}")

    @classmethod
    def delete_all_and_children(cls):
        # Delete child class records first
        for subclass in cls.__subclasses__():
            subclass.delete_all_and_children()

        # Delete records from the current class table
        if cls.table_name:
            cls._cursor.execute(f"DELETE FROM {cls.table_name}")

    @classmethod
    def where(cls, **kwargs):
        # Get the table names for the foreign key objects
        table_names = {key: getattr(value, "table_name", None) for key, value in kwargs.items()}

        # Create the conditions for the query
        conditions = []
        for key, value in kwargs.items():
            table = table_names[key] if table_names[key] else cls.table_name
            if table_names[key]:  # If it's a foreign key object
                conditions.append(f"{table}.id=?")
            else:  # If it's a normal attribute
                conditions.append(f"{cls.table_name}.{key}=?")

        # Get the values for the query
        values = [getattr(value, "id", value) for value in kwargs.values()]

        # Build the query
        joins = " ".join(
            f"INNER JOIN {table_name} ON {table_name}.id = {cls.table_name}.{key}_id"
            for key, table_name in table_names.items()
            if table_name is not None
        )
        conditions_str = " AND ".join(conditions)
        query = f"SELECT {cls.table_name}.* FROM {cls.table_name} {joins} WHERE {conditions_str}"

        # Execute the query
        cls._cursor.execute(query, tuple(values))
        return [cls(*row) for row in cls._cursor.fetchall()]