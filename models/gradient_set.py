import pickle
from typing import Any, List
from models.base_model import BaseModel
from models.motion import Motion
from models.patient import Patient
from datetime import datetime
import pandas as pd

from models.patient_motion import PatientMotion
from exp_motion_sample_trial import ExpMotionSampleTrial
from motion_filter import MotionFilter

class GradientSet(BaseModel):
    table_name = "gradient_set"

    def __init__(self, id=None, sensor_id=None, trial_id=None, matrix=None, conn=None, cursor=None, created_at=None,updated_at=None):
        super().__init__()
        self.id = id
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix

    # Splits the series based on zero value crossing.
    def get_sub_motions(self):
        if not self.sensor_id:
            return None
        from importlib import import_module
        Sensor = import_module("models.sensor").Sensor
        data = self.get_matrix("matrix")
        import pdb
        pdb.set_trace()
        name = Sensor.get(self.sensor_id).name
        est = ExpMotionSampleTrial(name, name, grad=data)

        return est.valid_sub_motions()


    def get_sensor_name(self):
        self._cursor.execute("""
            SELECT name FROM sensor
            WHERE id = ?
        """, (self.sensor_id,))
        
        row = self._cursor.fetchone()
        return row[0] if row else None


    def get_valid_motions(self):
        self.sub_motions = self.get_sub_motions()
        MotionFilter.get_valid_motions(self)

    def get_motion(self):
        self._cursor.execute("""
            SELECT motion.* FROM motion
            JOIN patient_motion ON motion.id = patient_motion.motion_id
            JOIN trial ON trial.patient_motion_id = patient_motion.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Motion(*row) if row else None

    def get_patient(self):
        self._cursor.execute("""
            SELECT patient.* FROM patient
            JOIN patient_motion ON patient.id = patient_motion.patient_id
            JOIN trial ON trial.patient_motion_id = patient_motion.id
            WHERE trial.id = ?
        """, (self.trial_id,))

        row = self._cursor.fetchone()
        return Patient(*row) if row else None

    # Splits the series based on zero value crossing.
    def split_series(self) -> Any:
        series = self.gradients[self.motions]
        split_indices = []

        for i in range(1, len(series)):
            if (series.iloc[i] > 0 and series.iloc[i - 1] < 0) or (series.iloc[i] < 0 and series.iloc[i - 1] > 0):
                split_indices.append(i)
        start = 0
        for end in split_indices:
            yield series[start:end]
            start = end

        return series[start:]

    def get_patient_motion_id(self):
        self._cursor.execute("SELECT patient_motion_id FROM gradient_set WHERE id=?", (self.id,))
        return self._cursor.fetchone()[0]

    def get_patient_motion(self):
        patient_motion_id = self.get_patient_motion_id()
        return PatientMotion.get(patient_motion_id)

    def mat(self):
        print(self.matrix)
        # Deserialize the 'matrix' value from the binary format using pickle
        return pd.Series(pickle.loads(self.matrix))

    def deserialize_matrix(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pickle
        
