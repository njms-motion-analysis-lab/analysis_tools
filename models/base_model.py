import sqlite3
conn = sqlite3.connect('motion_analysis.db')
cursor = conn.cursor()

class BaseModel:
    table_name = ""

    def create(self, **kwargs):
        keys = ", ".join(kwargs.keys())
        values = ", ".join("?" for _ in kwargs)
        try:
            cursor.execute(f"INSERT INTO {self.table_name} ({keys}) VALUES ({values})", tuple(kwargs.values()))
            conn.commit()
            self.id = cursor.lastrowid  # Assign the ID after successful insert
        except sqlite3.IntegrityError as e:
            print(f"Error creating record: {e}")
            return False
        return True

    def update(self, **kwargs):
        updates = ", ".join(f"{key}=?" for key in kwargs)
        try:
            cursor.execute(f"UPDATE {self.table_name} SET {updates} WHERE id=?", tuple(kwargs.values()) + (self.id,))
            conn.commit()
        except sqlite3.IntegrityError as e:
            print(f"Error updating record: {e}")
            return False
        return True

    def delete(self):
        cursor.execute(f"DELETE FROM {self.table_name} WHERE id=?", (self.id,))
        conn.commit()

    @classmethod
    def get(cls, id):
        cursor.execute(f"SELECT * FROM {cls.table_name} WHERE id=?", (id,))
        row = cursor.fetchone()
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
        cursor.execute(f"SELECT * FROM {cls.table_name} WHERE {conditions}", values)
        row = cursor.fetchone()

        if row:
            return cls(*row)
        else:
            # Create the record if not found
            cls_instance = cls(None, *values)
            cls_instance.create(**kwargs)
            cls_instance.id = cursor.lastrowid
            return cls_instance
    
    @classmethod
    def get_all(cls):
        cursor.execute(f"SELECT * FROM {cls.table_name}")
        return [cls(*row) for row in cursor.fetchall()]
