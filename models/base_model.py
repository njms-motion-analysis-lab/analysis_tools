from datetime import datetime
import pickle
import psycopg2
import pandas as pd

from database import Database


class BaseModel:
    db = Database.get_instance()
    _conn = db.connection
    _cursor = db.cursor

    def __init__(self, id=None, created_at=None, updated_at=None, **kwargs):
        self.id = id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

    def create(self, **kwargs):
        keys = ', '.join(['created_at', 'updated_at'] + list(kwargs.keys()))
        values = ', '.join(['%s', '%s'] + ['%s'] * len(kwargs))
        
        now = datetime.now()
        self.created_at = now
        self.updated_at = now
        try:
            self._cursor.execute(f"INSERT INTO {self.table_name} ({keys}) VALUES ({values}) RETURNING id", 
                                 (self.created_at, self.updated_at) + tuple(kwargs.values()))
            self.id = self._cursor.fetchone()[0]
            self._conn.commit()
            return self.id
        except psycopg2.IntegrityError as e:
            print(f"Error creating record: {e}")
            self._conn.rollback()

    @classmethod
    def table_exists(cls):
        cls._cursor.execute(f"SELECT to_regclass('public.{cls.table_name}');")
        return bool(cls._cursor.fetchone()[0])

    def update(self, **kwargs):
        if not self.__class__.table_exists():
            raise Exception(f"Table {self.table_name} does not exist.")
    
        updates = ", ".join(f"{key}=%s" for key in kwargs)
        updates += ", updated_at=%s"

        now = datetime.now()
        self.updated_at = now
        try:
            self._cursor.execute(f"UPDATE {self.table_name} SET {updates} WHERE id=%s", 
                                 tuple(kwargs.values()) + (self.updated_at, self.id,))
            self._conn.commit()
            
            for key, value in kwargs.items():
                setattr(self, key, value)
        except psycopg2.IntegrityError as e:
            print(f"Error updating record: {e}")
            self._conn.rollback()
            return False
        return True

    def delete(self):
        self._cursor.execute(f"DELETE FROM {self.table_name} WHERE id=%s", (self.id,))
        self._conn.commit()

    def get_matrix(self, column_name):
        self.__class__._cursor.execute(f"SELECT {column_name} FROM {self.table_name} WHERE id=%s", (self.id,))
        row = self.__class__._cursor.fetchone()

        if row:
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
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE id=%s", (id,))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return "nope"

    @classmethod
    def find_by(cls, column_name, value):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {column_name}=%s", (value,))
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
        conditions = " AND ".join(f"{key}=%s" for key in keys)
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {conditions}", values)
        row = cls._cursor.fetchone()
        if row:
            print(f"found {conditions}")
            return cls(*row)
        else:
            print("creating..")

            if 'matrix' in kwargs:
                kwargs["matrix"] = memoryview(pickle.dumps(old_matrix))
                
            cls_instance = cls()
            
            for key, value in kwargs.items():
                setattr(cls_instance, key, value)
            self_id = cls_instance.create(**kwargs)
            new_instance = cls.get(self_id)
            if new_instance.id is None:
                raise ValueError(f"Failed to create instance of {cls.__name__} with parameters: {kwargs}")
            
            return new_instance
    
    @classmethod
    def all(cls):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name}")
        return [cls(*row) for row in cls._cursor.fetchall()]

    @classmethod
    def last(cls, n):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} ORDER BY updated_at DESC LIMIT %s", (n,))
        rows = cls._cursor.fetchall()
        return [cls(*row) for row in rows]

    @classmethod
    def delete_all(cls):
        cls._cursor.execute(f"DELETE FROM {cls.table_name}")

    @classmethod
    def delete_all_and_children(cls):
        for subclass in cls.__subclasses__():
            subclass.delete_all_and_children()

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
                conditions.append(f"{table}.id=%s")
            else:  # If it's a normal attribute
                if isinstance(value, list):
                    placeholders = ', '.join(['%s' for _ in value])
                    conditions.append(f"{cls.table_name}.{key} IN ({placeholders})")
                    values.extend(value)
                else:
                    conditions.append(f"{cls.table_name}.{key}=%s")
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