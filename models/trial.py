from models.base_model import BaseModel
import sqlite3

from models.patient_motion import PatientMotion


class Trial(BaseModel):
    table_name = "trial"

    def __init__(self, id=None, patient_motion_id=None, timestamp=None):
        super().__init__()
        self.id = id
        self.patient_motion_id = patient_motion_id
        self.timestamp = timestamp

    def create(self, **kwargs):
        self.patient_motion_id = kwargs.get("patient_motion_id")
        self.timestamp = kwargs.get("timestamp")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_motion_id = kwargs.get("patient_motion_id", self.patient_motion_id)
        self.timestamp = kwargs.get("timestamp", self.timestamp)
        return super().update(**kwargs)

    def get_patient(self):
        from importlib import import_module
        Patient = import_module("models.motion").Patient
        patient_motion = PatientMotion.get(id=self.patient_motion_id)
        if patient_motion:
            return Patient.get(id=patient_motion.patient_id)
        return None

    def get_motion(self):
        from importlib import import_module
        Motion = import_module("models.motion").Motion
        patient_motion = PatientMotion.get(id=self.patient_motion_id)
        if patient_motion:
            return Motion.get(id=patient_motion.motion_id)
        return None