import pickle
from models.base_model import BaseModel
from models.motion import Motion
from models.patient import Patient
from datetime import datetime

from models.patient_motion import PatientMotion

class PositionSet(BaseModel):
    table_name = "position_set"

    def __init__(self, id=None, sensor_id=None, trial_id=None, matrix=None, conn=None, cursor=None, created_at=None,updated_at=None):
        super().__init__()
        self.id = id
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix

    def get_motion(self):
        self._cursor.execute("""
            SELECT motion.* FROM motion
            JOIN patient_motion ON motion.id = patient_motion.motion_id
            JOIN trial ON trial.patient_motion_id = patient_motion.id
            WHERE trial.id = ?
        """, (self.trial_id,))


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

    def get_patient_motion_id(self):
        self._cursor.execute("SELECT patient_motion_id FROM position_set WHERE id=?", (self.id,))
        return self._cursor.fetchone()[0]

    def get_patient_motion(self):
        patient_motion_id = self.get_patient_motion_id()
        return PatientMotion.get(patient_motion_id)

    def mat(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pd.Series(pickle.loads(self.matrix))

    def deserialize_matrix(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        return pickle.loads(self.matrix)