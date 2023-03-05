import numpy as np

class ExpMotion:
    def __init__(self, name: str) -> None:
        self.name = name
        self.exp_motion_sample_sets = []
        self.stdev = 0
        self.mean = 0

    def __str__(self) -> str:
        return self.name

    def add_sample(self, sample) -> None:
        self.exp_motion_sample_sets.append(sample)

    def set_stats(self) -> None:
        curr_collection = [sample_set.mean for sample_set in self.exp_motion_sample_sets]
        self.stdev = round(np.std(curr_collection), 2)
        self.mean = round(np.mean(curr_collection), 2)
