# sub_gradient.py
from models.base_model import BaseModel

import sqlite3

class SubGradient(BaseModel):
    table_name = "sub_gradient"

    def __init__(self, id=None, name=None, valid=None, matrix=None, gradient_set_id=None, gradient_set_ord=None, start_time=None, stop_time=None, mean=None, median=None, stdev=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.valid = valid
        self.matrix = matrix
        self.gradient_set_id = gradient_set_id
        self.gradient_set_ord = gradient_set_ord
        self.start_time = start_time
        self.stop_time = stop_time
        self.mean = mean
        self.median = median
        self.stdev = stdev
    
    def gradient_set(self):
        from models.gradient_set import GradientSet
        return GradientSet.get(id=self.gradient_set_id)

    def grad_matrix(self):
        parent_matrix = self.gradient_set().mat()
        return parent_matrix.loc[self.start_time:self.stop_time]

    def pos_matrix(self):
        from models.position_set import PositionSet
        parent_gradient_set = self.gradient_set()
        position_set = PositionSet.where(name=parent_gradient_set.name, trial_id=parent_gradient_set.trial_id, sensor_id=parent_gradient_set.sensor_id)[0]
        parent_position_matrix = position_set.mat()
        return parent_position_matrix.loc[self.start_time:self.stop_time]

