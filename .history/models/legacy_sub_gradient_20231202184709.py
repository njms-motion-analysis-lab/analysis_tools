# sub_gradient.py
from legacy_database import Database
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import skew
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

import sqlite3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Based on the call
FEATURE_EXTRACT_SETTINGS = {
    'variance_larger_than_standard_deviation': None,
    'has_duplicate_max': None,
    'has_duplicate_min': None,
    'has_duplicate': None,
    'sum_values': None,
    'abs_energy': None,
    'mean_abs_change': None,
    'mean_change': None,
    'mean_second_derivative_central': None,
    'median': None,
    'mean': None,
    'length': None,
    'standard_deviation': None,
    'variation_coefficient': None,
    'variance': None,
    'skewness': None,
    'kurtosis': None,
    'root_mean_square': None,
}


class SubGradient(LegacyBaseModel):
    table_name = "sub_gradient"

    def __init__(self, id=None, name=None, valid=None, matrix=None, gradient_set_id=None, gradient_set_ord=None, start_time=None, stop_time=None, mean=None, median=None, stdev=None, normalized=None, submovement_stats=None, submovement_stats_nonnorm=None, submovement_stats_position=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.valid = valid
        self.matrix = matrix # velocities
        self.gradient_set_id = gradient_set_id
        self.gradient_set_ord = gradient_set_ord
        self.start_time = start_time
        self.stop_time = stop_time
        self.mean = mean
        self.median = median
        self.stdev = stdev
        self.normalized = normalized
        self.submovement_stats = submovement_stats
        self.submovement_stats_nonnorm = submovement_stats_nonnorm
        self.submovement_stats_position = submovement_stats_position

    
    def gradient_set(self):
        from models.legacy_gradient_set import GradientSet
        return GradientSet.get(id=self.gradient_set_id)

    def grad_matrix(self):
        parent_matrix = self.gradient_set().mat()
        return parent_matrix.loc[self.start_time:self.stop_time]

    def pos_matrix(self):
        from models.legacy_position_set import PositionSet
        parent_gradient_set = self.gradient_set()
        position_set = PositionSet.where(name=parent_gradient_set.name, trial_id=parent_gradient_set.trial_id, sensor_id=parent_gradient_set.sensor_id)[0]
        parent_position_matrix = position_set.mat()
        return parent_position_matrix.loc[self.start_time:self.stop_time]

    def normalize(self):
        #normalize amplitude of submovement
        normed_amplitude = abs(self/np.max(np.abs(self)))
        start, end = normed_amplitude.index[0], normed_amplitude.index[-1]

        if end != start:
            x_vals = np.arange(start, end, (end - start) / 100).tolist()
        else:
            # Handle the case when end equals start
            x_vals = [start] * 100
        normed_temporally = np.interp(x_vals, normed_amplitude.index.tolist(), normed_amplitude)
        normed_temporally = pd.DataFrame({"grad_data":normed_temporally}, index=x_vals)
        normed_temporally["samplepoint"] = x_vals
        #print(normed_temporally)
        
        return memoryview(pickle.dumps(normed_temporally))

    def get_normalized(self):
        return pickle.loads(self.normalized)

    def get_matrix(self):
        return pickle.loads(self.matrix)

    def calc_sub_stats(self):
        motion = self.get_normalized()['grad_data']
        #print("MATRIX\n",pickle.loads(self.matrix))
        motionstats = {"mean": np.mean(motion), "median": np.median(motion), 
                        "sd":np.std(motion), "IQR": np.subtract(*np.percentile(motion, [75, 25])),
                        "RMS": np.sqrt(np.mean(motion**2)), "skewness": skew(motion),
                        "logmean_nonnormvelocity": np.log(np.mean(np.abs((self.matrix))))
                        }
            
            # ################# NEED TO DO ############# also get log absolute displacement
            #print(self.positional)

        ### ADD IN TSFRESH STATS HERE #####

        #print("we calculated the submovement stats\n", motionstats)
        #self.submovement_stats = memoryview(pickle.dumps(motionstats))
        return memoryview(pickle.dumps(motionstats))

    def get_sub_stats(self, normalized=True, abs_val=False):
        if normalized is True or normalized is 1:
            if self.submovement_stats is None:
                print("yoooooo")
                stats_needed = SubGradient.get_tsfresh_stats(self, abs_val=abs_val)
            else:
                stats_needed = self.submovement_stats
        else:
            if self.submovement_stats_nonnorm is None or abs_val is True or abs_val == 1:
                stats_needed = SubGradient.get_tsfresh_stats_non_normalized(self, abs_val=abs_val)
            else:
                stats_needed = self.submovement_stats_nonnorm
        return pickle.loads(stats_needed)

    def get_tsfresh_stats(self, normalized=True, abs_val=False):
        submovement = self.get_normalized()
        if abs_val is True or abs_val is 1:
            submovement = abs(submovement)
        submovement["id"] = self.id
        features = extract_features(submovement, column_id='id', column_sort='samplepoint', n_jobs=1, default_fc_parameters=ComprehensiveFCParameters())
        ans = memoryview(pickle.dumps(features))
        # self.update(submovement_stats=ans)
        return ans

    def get_tsfresh_stats_non_normalized(self, abs_val=False):
        submovement = pd.DataFrame(self.get_matrix())

        if abs_val is True or abs_val is 1:
            submovement = abs(submovement)
        submovement["id"] = self.id
        submovement["samplepoint"] = submovement.index.tolist()
        features = extract_features(submovement, column_id='id', column_sort='samplepoint', n_jobs=1, default_fc_parameters=ComprehensiveFCParameters())
        ans = memoryview(pickle.dumps(features))
        # self.update(submovement_stats_nonnorm=ans)
        return ans
    
    def get_tsfresh_stats_position(self, manual_position_data=None, default_fc_parameters=FEATURE_EXTRACT_SETTINGS):
        if manual_position_data is not None:
            positiondata = pd.DataFrame(manual_position_data)    
        else:
            positiondata = pd.DataFrame(self.pos_matrix())

        positiondata["id"] = self.id

        positiondata["samplepoint"] = positiondata.index.tolist()


        features = extract_features(positiondata, column_id='id', column_sort='samplepoint', n_jobs=1, default_fc_parameters=ComprehensiveFCParameters())

        return memoryview(pickle.dumps(features))
