from exp_motion_sample_trial import ExpMotionSampleTrial


class MotionFilter:
    DURATION = 10  # (ms)
    VELOCITY = 0.01  # (m/s)
    DISPLACEMENT = 0.05  # (cm)

    def __init__(self, exp_sample: ExpMotionSampleTrial, duration=DURATION, velocity=VELOCITY, displacement=DISPLACEMENT):
        self.exp_sample = exp_sample
        self.sub_motions = exp_sample.sub_motions
        self.duration = duration
        self.velocity = velocity
        self.displacement = displacement

    @classmethod
    def is_valid(cls, sub_motion, exp_sample):
        sub_motion_length = sub_motion.index.stop - sub_motion.index.start
        if sub_motion_length < cls.DURATION:
            return False
        elif abs(sub_motion.mean()) < cls.VELOCITY:
            return False

        range_start = sub_motion.index[0]
        range_end = sub_motion.index[-1]
        disp = abs(float(exp_sample.positional[exp_sample.motions].loc[range_start:range_end].max() - exp_sample.positional[exp_sample.motions].loc[range_start:range_end].min()))
        if disp < cls.DISPLACEMENT:
            return False
        return True

    @classmethod
    def get_valid_motions(cls, exp_sample):
        valid_motions = []
        for sub_motion in exp_sample.sub_motions:
            if cls.is_valid(sub_motion=sub_motion, exp_sample=exp_sample):
                valid_motions.append(sub_motion)
        return valid_motions