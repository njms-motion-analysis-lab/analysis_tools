# sub_gradient.py
from models.base_model import BaseModel
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy.stats import skew
from tsfresh import extract_features

import sqlite3

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class DynamicSubGradient(BaseModel):
    table_name = "dynamic_sub_gradient"

    def __init__(self, id=None, name=None, valid=None, matrix=None, dynamic_gradient_set_id=None, dynamic_gradient_set_ord=None, start_time=None, stop_time=None, normalized=None, submovement_stats=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.valid = valid
        self.matrix = matrix # velocities
        self.dynamic_gradient_set_id = dynamic_gradient_set_id
        self.dynamic_gradient_set_ord = dynamic_gradient_set_ord
        self.start_time = start_time
        self.stop_time = stop_time
        self.normalized = normalized
        self.submovement_stats = submovement_stats
    
    def dynamic_gradient_set(self):
        from models.dynamic_gradient_set import DynamicGradientSet
        return DynamicGradientSet.get(id=self.dynamic_gradient_set_id)
    
    def dynamic_gradient_set_df(self):
        # Deserialize the 'matrix' value from the binary format using pickle
        series = self.get_normalized()
        
        # Convert the pandas Series to a DataFrame
        dataframe = series.to_frame(name='value')
        
        # Reset the index and add a new column for the time points
        dataframe.reset_index(inplace=True)
        dataframe.rename(columns={'index': 'time'}, inplace=True)

        return dataframe

    def grad_matrix(self):
        parent_matrix = self.dynamic_gradient_set().mat()
        return parent_matrix.loc[self.start_time:self.stop_time]

    def pos_matrix(self):
        from models.position_set import PositionSet
        parent_dynamic_gradient_set = self.dynamic_gradient_set()
        position_set = PositionSet.where(name=parent_dynamic_gradient_set.name, trial_id=parent_dynamic_gradient_set.trial_id, sensor_id=parent_dynamic_gradient_set.sensor_id)[0]
        parent_position_matrix = position_set.mat()
        return parent_position_matrix.loc[self.start_time:self.stop_time]

    def normalize(self):
        #normalize amplitude of submovement
        normed_amplitude = abs(self/np.max(np.abs(self)))
        start, end = normed_amplitude.index[0], normed_amplitude.index[-1]
        x_vals = np.arange(start,end,(end-start)/100).tolist()
        normed_temporally = np.interp(x_vals, normed_amplitude.index.tolist(), normed_amplitude)
        normed_temporally = pd.DataFrame(normed_temporally, index=x_vals)
        #print(normed_temporally)
        
        return memoryview(pickle.dumps(normed_temporally))

    def get_normalized(self):
        return pickle.loads(self.normalized)

    def calc_sub_stats(self):
        if isinstance(self.matrix, bytes):
            self.matrix = memoryview(self.matrix)
            
        matrix_array = pickle.loads(self.matrix)
        motion = self.get_normalized()[0]

        motionstats = {"mean": np.mean(motion), "median": np.median(motion),
                    "sd": np.std(motion), "IQR": np.subtract(*np.percentile(motion, [75, 25])),
                    "RMS": np.sqrt(np.mean(motion**2)), "skewness": skew(motion),
                    "logmean_nonnormvelocity": np.log(np.mean(np.abs((matrix_array))))
                    }

        return memoryview(pickle.dumps(motionstats))

    def get_sub_stats(self):
        return pickle.loads(self.submovement_stats)

    def get_tsfresh_data(self):
        matrix_df = self.dynamic_gradient_set_df()
        matrix_df['id'] = 0
        features = extract_features(matrix_df, column_id='id', column_sort='time')
        return features.info()