from models.base_model import BaseModel
from models.motion import Motion
from models.patient import Patient
from models.trial import Trial

class PositionSet(BaseModel):
    table_name = "position_set"

    def __init__(self, id=None, sensor_id=None, trial_id=None, matrix=None, conn=None, cursor=None):
        super().__init__()
        self.id = id
        self.sensor_id = sensor_id
        self.trial_id = trial_id
        self.matrix = matrix
        self._conn = conn
        self._cursor = cursor

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