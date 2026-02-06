from chart_review_scoliosis_time_predictor import ScoliosisTimePredictor
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep, proportion_effectsize
from statsmodels.stats.power import NormalIndPower
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from catboost import CatBoostClassifier


# Configure classification models with proper balancing
CLASSIFICATION_MODELS = {
    "RandomForestClassifier": {
        "classifier": RandomForestClassifier(class_weight='balanced'),
        "param_grid": {
            "classifier__n_estimators": [50, 150],
            "classifier__max_depth": [1, 3, 5, 7],
            "classifier__min_samples_split": [2, 3],
            "classifier__min_samples_leaf": [1, 2, 3],
            "classifier__max_features": ["sqrt", "log2"]
        }
    },
    # "LogisticRegression": {
    #     # Penalized Logistic Regression (L1 or L2)
    #     "classifier": LogisticRegression(class_weight='balanced', max_iter=1000),
    #     "param_grid": {
    #         "classifier__C": [0.01, 0.1, 1, 10],
    #         "classifier__penalty": ["l1", "l2"],
    #         # 'liblinear' supports both 'l1' and 'l2'. 
    #         # If you want to try 'saga', add it to the solver list.
    #         "classifier__solver": ["liblinear"]
    #     }
    # },
    # "XGBClassifier": {
    #     "classifier": XGBClassifier(random_state=42, eval_metric='logloss'),
    #     "param_grid": {
    #         "classifier__n_estimators": [100, 200],
    #         "classifier__max_depth": [3, 6],
    #         "classifier__learning_rate": [0.01, 0.1],
    #         "classifier__scale_pos_weight": [1, 5, 10]  # Adjust dynamically if needed
    #     }
    # # },
    "CatBoostClassifier": {
        "classifier": CatBoostClassifier(verbose=False, random_state=42),
        "param_grid": {
            "classifier__iterations": [100, 200],
            "classifier__depth": [3, 5, 7],
            "classifier__learning_rate": [0.01, 0.1]
        }
    }
}

# Update folder paths to reflect new structure
RAW_DATA_FOLDER = "scoliosis_chart_review/raw_chart_data"  # now located at scoliosis_chart_review/raw_data
RESULTS_FOLDER = "results"    # output files will be stored here

# Create the results folder if it doesn't exist
os.makedirs(RESULTS_FOLDER, exist_ok=True)

for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    for filename in filenames:
        # Initialize predictor with proper CSV path
        csv_path = os.path.join(subdir, filename)
        stp = ScoliosisTimePredictor(csv_path=csv_path)
        
        # Generate and filter dataframe
        TARGET = "tothlos"
        print("XXX", filename)
        df = stp.generate_training_dataframe(target_col=TARGET)


        
        # =============================================================================
        # UPDATED: Instead of keeping columns that start with "abx_regimen", we now
        # keep those that begin with either "combo_abx_regimen_" or "single_abx_regimen_"
        # =============================================================================
        import pdb;pdb.set_trace()

        best_pipeline, best_metrics, best_model_name = stp.grid_search_pipeline(data=df, target_column=TARGET, models=CLASSIFICATION_MODELS, bin_string="x > 5")

        import pdb;pdb.set_trace()