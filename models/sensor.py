from models.base_model import BaseModel
from models.gradient_set import GradientSet
from models.position_set import PositionSet

class Sensor(BaseModel):
    table_name = "sensor"
    _conn = BaseModel._conn
    _cursor = BaseModel._cursor

    def __init__(self, id=None, name=None, axis=None, part=None, side=None, iteration=None, conn=None, cursor=None, kind="position"):
        super().__init__()
        self.id = id
        self.name = name
        self.axis = axis
        self.part = part
        self.side = side
        self.iteration = iteration
        self.kind = kind


    def get_position_sets(self, motion):
        self._cursor.execute("""
            SELECT position_set.* FROM position_set
            JOIN trial ON position_set.trial_id = trial.id
            JOIN patient_motion ON trial.patient_motion_id = patient_motion.id
            WHERE position_set.sensor_id = ? AND patient_motion.motion_id = ?
        """, (self.id, motion.id))

        return [PositionSet(*row) for row in self._cursor.fetchall()]

    def get_gradient_sets(self, motion):
        self._cursor.execute("""
            SELECT gradient_set.* FROM gradient_set
            JOIN trial ON gradient_set.trial_id = trial.id
            JOIN patient_motion ON trial.patient_motion_id = patient_motion.id
            WHERE gradient_set.sensor_id = ? AND patient_motion.motion_id = ?
        """, (self.id, motion.id))

        return [GradientSet(*row) for row in self._cursor.fetchall()]


    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()