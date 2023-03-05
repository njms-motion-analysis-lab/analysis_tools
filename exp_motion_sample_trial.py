import constants
import numpy as np
from typing import Any

class ExpMotionSampleTrial:
    FILE_PATH = "/Users/stephenmacneille/Documents/labs/NPYViewer/sample_npy_files/filename.npy"

    def __init__(self, name: str, measurements: dict, motions: str) -> None:
        self.name = name
        self.motions = motions
        self.gradients = measurements[constants.GRADIENTS]
        self.positional = measurements[constants.POSITIONAL]       
        self.sub_motions = list(self.split_series())

    # Splits the series based on zero value crossing.
    def split_series(self) -> Any:
        series = self.gradients[self.motions]
        split_indices = []

        for i in range(1, len(series)):
            if (series.iloc[i] > 0 and series.iloc[i - 1] < 0) or (series.iloc[i] < 0 and series.iloc[i - 1] > 0):
                split_indices.append(i)
        start = 0
        for end in split_indices:
            yield series[start:end]
            start = end

        return series[start:]
    
    # Gets the other axes for a provided coordinate.
    def letter_substitution(self) -> Any:
        string = self.motions
        result = {'original': string}
        last_char = string[-1]

        if last_char == 'x':
            result['substituted_y'] = string[:-1] + 'y'
            result['substituted_z'] = string[:-1] + 'z'
        elif last_char == 'y':
            result['substituted_x'] = string[:-1] + 'x'
            result['substituted_z'] = string[:-1] + 'z'
        elif last_char == 'z':
            result['substituted_x'] = string[:-1] + 'x'
            result['substituted_y'] = string[:-1] + 'y'
    
        return result.values()

    def get_axes(self, sub_motion, axis_type) -> Any:
        other_headers = self.letter_substitution()
        range_start = sub_motion.index[0]
        range_end = sub_motion.index[-1]
        if axis_type == "grad":
            return self.gradients[other_headers].loc[range_start:range_end]
        elif axis_type == "pos":
            return self.positional[other_headers].loc[range_start:range_end]

    def save_axes(self, sub_motion, axis_type) -> None:
        np.save(self.FILE_PATH, self.get_axes(sub_motion, axis_type).values)