from datetime import datetime
import pickle
import sqlite3
import pandas as pd

from legacy_database import Database


class BaseModel:
    db = Database.get_instance()
    _conn = db.connection
    _cursor = db.cursor


    def select(objects, **kwargs):
        def match(obj):
            return all(getattr(obj, key) == value for key, value in kwargs.items())
        
        return [obj for obj in objects if match(obj)]

    @classmethod
    def class_sort_by(cls, attribute):
        if not cls.table_exists():
            raise Exception(f"Table {cls.table_name} does not exist.")

        cls._cursor.execute(f"SELECT * FROM {cls.table_name} ORDER BY {attribute} ASC")
        rows = cls._cursor.fetchall()

        if rows:
            return [cls(*row) for row in rows]
        else:
            return None
    
    @classmethod
    def sort_by(cls, instances, attribute):
        return sorted(instances, key=lambda instance: getattr(instance, attribute))

    def __init__(self, id=None, created_at=None, updated_at=None, **kwargs):
        self.id = id
        self.created_at = created_at if created_at else datetime.now()
        self.updated_at = updated_at if updated_at else datetime.now()
    
    @classmethod
    def set_class_connection(cls, test_mode=False):
        if test_mode is True:
            cls.db.connection = sqlite3.connect('motion_analysis_test.db')
            cls.db.cursor = cls.db.connection.cursor()

    def set_connection(self, test_mode=False):
        if test_mode is True:
            self.__class__.db.connection = sqlite3.connect('motion_analysis_test.db')
            self.__class__.db.cursor = self.__class__.db.connection.cursor()

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

    @classmethod
    def table_exists(cls):
        cls._cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{cls.table_name}';")
        return bool(cls._cursor.fetchone())

    def update(self, **kwargs):
        if not self.__class__.table_exists():
            raise Exception(f"Table {self.table_name} does not exist.")
    
        updates = ", ".join(f"{key}=?" for key in kwargs)
        updates += ", updated_at=?"  # Automatically update the updated_at timestamp

        now = datetime.now()
        self.updated_at = now

        sql_query = f"UPDATE {self.table_name} SET {updates} WHERE id=?"
        sql_params = tuple(kwargs.values()) + (self.updated_at, self.id)

        try:
            print(f"Executing SQL Query: {sql_query} with params: {sql_params}")
            self.__class__._cursor.execute(sql_query, sql_params)
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

    def delete_self_and_children(self):
        # Retrieve and delete child instances first
        for subclass in self.__class__.__subclasses__():
            child_instances = subclass.where(**{f"{self.table_name}_id": self.id})
            for instance in child_instances:
                instance.delete_self_and_children()
        
        # Delete the current instance
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
            print(f"{attribute}: {str(value)[:200]}")

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
            print(f"found {conditions}")
            print(*row)
            return cls(*row)
        else:
            print("creating..")

            if 'matrix' in kwargs:
                kwargs["matrix"] = memoryview(pickle.dumps(old_matrix))
                # Serialize the matrix
                # Create the record if not found
                
            cls_instance = cls()
            
            for key, value in kwargs.items():
                setattr(cls_instance, key, value)
            cls_instance.id = cls_instance.create(**kwargs)
            
            if cls_instance.id is None:
                raise ValueError(f"Failed to create instance of {cls.__name__} with parameters: {kwargs}")
            
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
        values = []
        for key, value in kwargs.items():
            table = table_names[key] if table_names[key] else cls.table_name
            if table_names[key]:  # If it's a foreign key object
                conditions.append(f"{table}.id=?")
            else:  # If it's a normal attribute
                if isinstance(value, list):
                    conditions.append(f"{cls.table_name}.{key} IN ({','.join(['?']*len(value))})")
                    values.extend(value)
                else:
                    conditions.append(f"{cls.table_name}.{key}=?")
                    values.append(value)

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
    