import os
import numpy as np
from motion_filter import MotionFilter
from exp_motion_sample_trial import ExpMotionSampleTrial


class ExpMotionSample:
    def __init__(self, filename: str, motions: str):
        self.data = np.load(filename, allow_pickle=True).tolist()
        self.name = self.get_name(os.path.splitext(filename)[0])
        self.samples = self.data.keys()
        self.exp_samples = []
        self.valid_samples = []
        self.motions = motions
        self.mean = 0
        self.stdev = 0
        self.set_samples()
        self.set_stats()
    
    def set_samples(self) -> None:
        for sample in self.samples:
            new_sample = ExpMotionSampleTrial(sample, self.data[sample], self.motions)
            valid_sample = MotionFilter.get_valid_motions(new_sample)

            self.exp_samples.append(new_sample)
            self.valid_samples.append(valid_sample)
    
    def set_stats(self) -> None:
        curr_collection = []
        for valid_sample in self.valid_samples:
            curr_collection.append(len(valid_sample))
        self.stdev = round(np.std(curr_collection),2)
        self.mean = round(np.mean(curr_collection), 2)

    def get_name(self, ext) -> str:
        return ext.split(os.path.sep)[-1]
    
    def __str__(self):
        return self.name