from models.base_model import BaseModel
import sqlite3


class PatientMotion(BaseModel):
    table_name = "patient_motion"

    def __init__(self, id=None, patient_id=None, motion_id=None, created_at=None, updated_at=None):
        self.id = id
        self.patient_id = patient_id
        self.motion_id = motion_id
        self.created_at = created_at
        self.updated_at = updated_at

    def create(self, **kwargs):
        self.patient_id = kwargs.get("patient_id")
        self.motion_id = kwargs.get("motion_id")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_id = kwargs.get("patient_id", self.patient_id)
        self.motion_id = kwargs.get("motion_id", self.motion_id)
        return super().update(**kwargs)

    def add_trial(self, trial):
        """
        Add a Trial to the PatientMotion instance.

        :param trial: The Trial instance to be added.
        :return: None
        """
        if not hasattr(self, 'trials'):
            self.trials = []

        self.trials.append(trial)

        trial.patient_motion_id = self.id
        trial.update(patient_motion_id=self.id)

    @classmethod
    def get(cls, patient, motion):
        cls._cursor.execute("SELECT * FROM patient_motion WHERE patient_id=? AND motion_id=?", (patient.id, motion.id))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return None

    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()