from models.base_model import BaseModel
import sqlite3


class PatientMotion(BaseModel):
    table_name = "patient_motion"

    def __init__(self, id=None, patient_id=None, motion_id=None):
        self.id = id
        self.patient_id = patient_id
        self.motion_id = motion_id

    def create(self, **kwargs):
        self.patient_id = kwargs.get("patient_id")
        self.motion_id = kwargs.get("motion_id")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_id = kwargs.get("patient_id", self.patient_id)
        self.motion_id = kwargs.get("motion_id", self.motion_id)
        return super().update(**kwargs)

    @classmethod
    def get(cls, patient, motion):
        cls._cursor.execute("SELECT * FROM patient_motion WHERE patient_id=? AND motion_id=?", (patient.id, motion.id))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return None