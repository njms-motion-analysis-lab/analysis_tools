GRADIENTS: str = 'gradients'
POSITIONAL: str = 'positional'


class Wrist:
    def name() -> str:
        'wrist'

    class Position:
        class Left:
            class SensorA:
                DIMENSIONS = {
                    'X':'lwra_x',
                    'Y':'lwra_y',
                    'Z':'lwra_z',
                }

            class SensorB:
                DIMENSIONS = {
                    'X':'lwrb_x',
                    'Y':'lwrb_y',
                    'Z':'lwrb_z',
                }
                
                    
        class Right:
            class SensorA:
                DIMENSIONS = {
                    'X':'rwra_x',
                    'Y':'rwra_y',
                    'Z':'rwra_z',
                }

            class SensorB:
                DIMENSIONS = {
                    'X':'rwrb_x',
                    'Y':'rwrb_y',
                    'Z':'rwrb_z',
                }




PARTS = {
    'wrist': 'Wrist'
}

SIDES = {
    'r' : 'Right',
    'l' : 'Left',
}

SENSORS = {
    'a': 'SensorA',
    'b': 'SensorB',
}

ALLOWED_SENSORS = [
    "lwra_x",
    "lwrb_x",
    "lwra_y",
    "lwrb_y",
    "lwra_z",
    "lwrb_z",
    "rwra_x",
    "rwrb_x",
    "rwra_y",
    "rwrb_y",
    "rwra_z",
    "rwrb_z",
    "rfrm_x",
    "rfrm_y",
    "rfrm_z",
    "lelb_x",
    "lelb_y",
    "lelb_z",
    "relb_x",
    "relb_y",
    "relb_z",
    "lfrm_x",
    "lfrm_y",
    "lfrm_z",
]




TS_FRESH_SLIM = {
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
    'absolute_sum_of_changes': None,
    'longest_strike_below_mean': None,
    'longest_strike_above_mean': None,
    'count_above_mean': None,
    'count_below_mean': None,
    'last_location_of_maximum': None,
    'first_location_of_maximum': None,
    'last_location_of_minimum': None,
    'first_location_of_minimum': None,
    'percentage_of_reoccurring_values_to_all_values': None,
    'percentage_of_reoccurring_datapoints_to_all_datapoints': None,
    'sum_of_reoccurring_values': None,
    'sum_of_reoccurring_data_points': None,
    'ratio_value_number_to_time_series_length': None,
    'maximum': None,
    'absolute_maximum': None,
    'minimum': None,
}

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