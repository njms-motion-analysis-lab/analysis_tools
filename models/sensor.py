from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.base_model import BaseModel
from models.gradient_set import GradientSet
from models.position_set import PositionSet

class Sensor(BaseModel):
    table_name = "sensor"
    _conn = BaseModel._conn
    _cursor = BaseModel._cursor

    def __init__(self, id=None, name=None, axis=None, part=None, side=None, placement=None, conn=None, cursor=None, kind="position"):
        super().__init__()
        self.id = id
        self.name = name
        self.axis = axis
        self.part = part
        self.side = side
        self.placement = placement
        self.kind = kind


    def position_sets(self, task):
        self._cursor.execute("""
            SELECT position_set.* FROM position_set
            JOIN trial ON position_set.trial_id = trial.id
            JOIN patient_task ON trial.patient_task_id = patient_task.id
            WHERE position_set.sensor_id = ? AND patient_task.task_id = ?
        """, (self.id, task.id))

        return [PositionSet(*row) for row in self._cursor.fetchall()]

    def gradient_sets(self, task):
        self._cursor.execute("""
            SELECT gradient_set.* FROM gradient_set
            JOIN trial ON gradient_set.trial_id = trial.id
            JOIN patient_task ON trial.patient_task_id = patient_task.id
            WHERE gradient_set.sensor_id = ? AND patient_task.task_id = ?
        """, (self.id, task.id))

        return [GradientSet(*row) for row in self._cursor.fetchall()]

    def add_position_set(self, position_set):
        if position_set.sensor_id == self.id:
            print("This PositionSet is already associated with this sensor.")
            return

        position_set.sensor_id = self.id
        position_set.update(sensor_id=self.id)
        print(f"PositionSet with ID {position_set.id} has been associated with this sensor.")

    def add_gradient_set(self, gradient_set):
        if gradient_set.sensor_id == self.id:
            print("This GradientSet is already associated with this sensor.")
            return

        gradient_set.sensor_id = self.id
        gradient_set.update(sensor_id=self.id)
        print(f"GradientSet with ID {gradient_set.id} has been associated with this sensor.")

    @classmethod
    def delete_all(cls):
        cls.delete_all_and_children()