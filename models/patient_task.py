from models.base_model import BaseModel
import sqlite3


class PatientTask(BaseModel):
    table_name = "patient_task"

    def __init__(self, id=None, patient_id=None, task_id=None, created_at=None, updated_at=None):
        self.id = id
        self.patient_id = patient_id
        self.task_id = task_id
        self.created_at = created_at
        self.updated_at = updated_at

    def create(self, **kwargs):
        self.patient_id = kwargs.get("patient_id")
        self.task_id = kwargs.get("task_id")
        return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_id = kwargs.get("patient_id", self.patient_id)
        self.task_id = kwargs.get("task_id", self.task_id)
        return super().update(**kwargs)

    def add_trial(self, trial):
        """
        Add a Trial to the PatientTask instance.

        :param trial: The Trial instance to be added.
        :return: None
        """
        if not hasattr(self, 'trials'):
            self.trials = []

        self.trials.append(trial)

        trial.patient_task_id = self.id
        trial.update(patient_task_id=self.id)

    @classmethod
    def get(cls, patient, task):
        cls._cursor.execute("SELECT * FROM patient_task WHERE patient_id=? AND task_id=?", (patient.id, task.id))
        row = cls._cursor.fetchone()
        if row:
            return cls(*row)
        return None

    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()