import pickle
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_task import Task
from models.legacy_patient import Patient
from datetime import datetime
import pandas as pd

from models.legacy_patient_task import PatientTask

class PositionSet(LegacyBaseModel):
    table_name = "position_set"

    def __init__(self, id=None, name=None, sensor_id=None, trial_id=None, matrix=None, conn=None, cursor=None, created_at=None,updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix
    
    # def get_contralateral_position(self):
    #     sensor_name = Sensor.get(self.sensor_id).name
    #     opposite_name = Task.get_counterpart_sensor(sensor_name)



    def get_task(self):
        self._cursor.execute("""
            SELECT task.* FROM task
            JOIN patient_task ON task.id = patient_task.task_id
            JOIN trial ON trial.patient_task_id = patient_task.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Task(*row) if row else None

    def get_patient(self):
        self._cursor.execute("""
            SELECT patient.* FROM patient
            JOIN patient_task ON patient.id = patient_task.patient_id
            JOIN trial ON trial.patient_task_id = patient_task.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Patient(*row) if row else None


    def add_sensor(self, sensor):
        if self.sensor_id == sensor.id:
            print("This PositionSet is already associated with the provided sensor.")
            return

        self.sensor_id = sensor.id
        self.update(sensor_id=self.sensor_id)
        print(f"Sensor with ID {sensor.id} has been associated with this PositionSet.")


    def get_patient_task_id(self):
        self._cursor.execute("SELECT patient_task_id FROM position_set WHERE id=?", (self.id,))
        return self._cursor.fetchone()[0]

    def get_patient_task(self):
        patient_task_id = self.get_patient_task_id()
        return PatientTask.get(patient_task_id)

    def mat(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pd.Series(pickle.loads(self.matrix))

    def deserialize_matrix(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pickle.loads(self.matrix)