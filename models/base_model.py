import sqlite3


class BaseModel:
    table_name = ""

    @classmethod
    def set_connection(cls, test_mode=False, conn=None, cursor=None):
        if test_mode is True:
            cls._conn = sqlite3.connect('motion_analysis_test.db')
            cls._cursor = cls._conn.cursor()
        if (conn is not None) and (cursor is not None):
            cls._conn = conn
            cls._cursor = cursor


    def create(self, **kwargs):
        keys = ', '.join(kwargs.keys())
        values = ', '.join(['?'] * len(kwargs))
        print(keys)
        print(values)
        try:
            self._cursor.execute(f"INSERT INTO {self.table_name} ({keys}) VALUES ({values})", tuple(kwargs.values()))
            self.id = self._cursor.lastrowid
            self._conn.commit()
            return self.id
        except sqlite3.IntegrityError:
            return False

    def update(self, **kwargs):
        updates = ", ".join(f"{key}=?" for key in kwargs)
        try:
            self._cursor.execute(f"UPDATE {self.table_name} SET {updates} WHERE id=?", tuple(kwargs.values()) + (self.id,))
            self._conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Error updating record: {e}")
            return False
        return True

    def delete(self):
        self._cursor.execute(f"DELETE FROM {self.table_name} WHERE id=?", (self.id,))
        self._conn.commit()

    @classmethod
    def get(cls, id):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE id=?", (id,))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return None

    @classmethod
    def find_or_create(cls, **kwargs):
        """
        Find or create a record based on the provided attribute(s).

        Usage:
        m = Motion.find_or_create(description="Jumping")
        p = Patient.find_or_create(name="S001")

        Input:
        - kwargs: keyword arguments representing the attributes to search or create a record with.
        """
        keys = kwargs.keys()
        values = tuple(kwargs.values())

        # Find the record with the given attribute(s)
        conditions = " AND ".join(f"{key}=?" for key in keys)
        cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {conditions}", values)
        row = cls._cursor.fetchone()

        if row:
            return cls(*row)
        else:
            # Create the record if not found
            cls_instance = cls(*values)
            cls_instance.create(**kwargs)
            cls_instance.id = cls._cursor.lastrowid
            return cls_instance
    
    @classmethod
    def get_all(cls):
        cls._cursor.execute(f"SELECT * FROM {cls.table_name}")
        return [cls(*row) for row in cls._cursor.fetchall()]

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
