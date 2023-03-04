import numpy as np

class Patient:
    def __init__(self, name: str,) -> None:
        self.name = name
        self.exp_motions = []
        self.exp_motions_hash = {}
        self.mean = 0
        self.stdev = 0
    

    def add_exp_motions(self, motion) -> None:
        self.exp_motions.append(motion)
        split_name = motion.name.split('_')
        exp_motion = split_name[2]
        varient = split_name[-1]

        if varient != exp_motion:
            exp_motion = exp_motion + '_' + varient
        if exp_motion.endswith('_'):
            exp_motion = exp_motion[:-1]


        self.exp_motions_hash[exp_motion] = motion

    def set_stats(self) -> None:
        curr_collection = []
        for sample_set in self.exp_motions:
            curr_collection.append(sample_set.mean)
        self.stdev = round(np.std(curr_collection),2)
        self.mean = round(np.mean(curr_collection), 2)