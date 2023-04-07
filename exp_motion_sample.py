import os
import numpy as np
from motion_filter import MotionFilter
from exp_motion_sample_trial import ExpMotionSampleTrial


class ExpMotionSample:
    def __init__(self, filename: str, motions: str):
        self.data = np.load(filename, allow_pickle=True).tolist()
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.samples = self.data.keys()
        self.exp_samples = []
        self.valid_samples = []
        self.motions = motions
        self.mean = 0
        self.stdev = 0
        self.set_samples()
        self.set_stats()
    
    def set_samples(self):
        for sample in self.samples:
            import pdb
            pdb.set_trace()
            new_sample = ExpMotionSampleTrial(sample, self.data[sample], self.motions)
            valid_sample = MotionFilter.get_valid_motions(new_sample)

            self.exp_samples.append(new_sample)
            self.valid_samples.append(valid_sample)
    
    def set_stats(self):
        curr_collection = [len(valid_sample) for valid_sample in self.valid_samples]
        self.stdev = round(np.std(curr_collection), 2)
        self.mean = round(np.mean(curr_collection), 2)

    def __str__(self):
        return self.name