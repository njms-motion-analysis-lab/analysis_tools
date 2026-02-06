
import pickle
import re
import pandas as pd
from datetime import datetime
from prediction_tools.time_predictor import TimePredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
import np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix)
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
KNOWN_COMPOSITES = {
    "any_ssi": ["dsupinfec", "wndinfd", "orgspcssi", "dorgspcssi"],
    "any_reop": ["reoperation", "retorrelated", "reoperation2"]  # example
}
POTENTIAL_BINARY = [
    # From your original list (still valid)
    "sex",                  # "male"/"female" -> 1/0
    "ethnicity_hispanic",   # "yes"/"no"
    "transfus",             # "yes"/"no"
    "inout",                # "inpatient"/"outpatient"
    "dnr",                  # "yes"/"no"
    "prem_birth",           # "yes"/"no"
    "ventilat",             # "yes"/"no"
    "asthma",               # "yes"/"no"
    "oxygen_sup",           # "yes"/"no"
    "tracheostomy",         # "yes"/"no"
    "stillinhosp",          # "yes"/"no"
    "death30yn",            # "yes"/"no"
    "oxygen_at_discharge",  # "yes"/"no"

    # Additional columns that often appear as yes/no in your data snippet
    "malignancy",           # "yes"/"no"
    "nutr_support",         # "yes"/"no"
    "prsepis",              # "yes"/"no" (prior sepsis)
    "inotr_support",        # "yes"/"no" (inotrope support)
    "cpr_prior_surg",       # "yes"/"no"
    "preop_covid",          # "yes"/"no"
    "postop_covid",         # "yes"/"no"

    # Pediatric spinal specifics that are often yes/no
    "ped_sap_infection",    # prophylaxis infection
    "ped_sap_prophylaxis",  
    "ped_sap_redosed",      
    "ped_spn_antibio_wnd",  
    "ped_spn_antifib",      
    "ped_spn_trnsvol_cell", 
    "ped_spn_trnsvol_allogen",
    "ped_spn_post_trnsvol_cell",
    "ped_spn_post_trnsvol_allogen",
    
    # … add any other columns you know are “yes/no” fields.
]

class ScoliosisTimePredictor(TimePredictor):
    table_name = "scoliosis_time_predictor"

    def __init__(
        self,
        id=None,
        task_id=None,
        metrics=None,
        matrix=None,
        csv_path="raw_data/v3filteredaiscases.csv",
        created_at=None,
        updated_at=None,
        cohort_id=None,
        aggregated_stats=None,
        aggregated_stats_non_normed=None,
        **kwargs,
    ):
        """
        Initializes the ScoliosisTimePredictor object.
        """
        super().__init__(**kwargs)
        self.id = id
        self.task_id = task_id
        self.metrics = metrics
        self.matrix = matrix
        self.csv_path = csv_path
        self.created_at = created_at or datetime.now().isoformat()
        self.updated_at = updated_at or datetime.now().isoformat()
        self.cohort_id = cohort_id
        self.aggregated_stats = aggregated_stats
        self.aggregated_stats_non_normed = aggregated_stats_non_normed

    def prepare_train_test(self, data, target_column, test_size=0.2, random_state=42):
        """
        Splits the data into training and testing sets.
        
        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :param test_size: Proportion of the data to use as the test set.
        :param random_state: Random seed for reproducibility.
        :return: X_train, X_test, y_train, y_test
        """
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    def grid_search_pipeline(
        self, 
        data, 
        target_column="any_ssi",  # Changed default to your target
        test_size=0.2, 
        random_state=42, 
        models=None,  # Changed parameter name from regression_models
        cv_strategy=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ):
        """
        Performs GridSearchCV over classification models with clinical analysis enhancements.
        Returns best model and evaluation metrics.
        """
        if models is None or not isinstance(models, dict):
            raise ValueError("You must provide a valid dictionary of classification models and param grids.")

        # Split data with stratification
        X_train, X_test, y_train, y_test = self.prepare_train_test(
            data, target_column, test_size, random_state
        )

        best_estimator = None
        best_score = -np.inf
        best_model_name = None
        best_metrics = {}
        
        # Calculate class weights for metrics
        positive_prob = y_train.mean()

        for model_name, model_info in models.items():
            print(f"\n--- Grid Search for {model_name} ---")
            classifier = model_info["classifier"]
            param_grid = model_info["param_grid"]

            # Create pipeline with potential preprocessing
            pipeline = Pipeline([
                ("classifier", classifier)  # Changed from "regressor"
            ])

            # Configure scoring for clinical relevance
            scoring = {
                'auc': 'roc_auc',
                'accuracy': 'accuracy',
                'f1': 'f1',
                'precision': 'precision',
                'recall': 'recall'
            }

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring=scoring,
                refit='auc',  # Primary metric for model selection
                cv=cv_strategy,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            # Predict probabilities for clinical analysis
            y_pred = best_model.predict(X_test)
            y_proba = best_model.predict_proba(X_test)[:, 1]

            # Comprehensive metrics
            metrics = {
                'auc': roc_auc_score(y_test, y_proba),
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'best_params': grid_search.best_params_,
                'feature_importances': best_model.named_steps['classifier'].feature_importances_ 
                                    if hasattr(best_model.named_steps['classifier'], 'feature_importances_') 
                                    else None
            }

            print(f"\nBest {model_name} Test Performance:")
            print(f"AUC: {metrics['auc']:.3f} | F1: {metrics['f1']:.3f}")
            print(f"Precision: {metrics['precision']:.3f} | Recall: {metrics['recall']:.3f}")

            # Update best model based on AUC
            if metrics['auc'] > best_score:
                best_score = metrics['auc']
                best_estimator = best_model
                best_model_name = model_name
                best_metrics = metrics

        print(f"\n=== Best Overall Model: {best_model_name} ===")
        print(f"AUC: {best_metrics['auc']:.3f}")
        print(f"Confusion Matrix:\n{best_metrics['confusion_matrix']}")
        
        return best_estimator, best_metrics, best_model_name
    
    def run_random_forest_regression_with_shap(self, X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
        """
        Fits a Random Forest Regressor, evaluates it on the test set, and displays SHAP beeswarm plot.

        :param X_train: Training features.
        :param X_test: Testing features.
        :param y_train: Training target.
        :param y_test: Testing target.
        :param n_estimators: Number of trees in the Random Forest.
        :param random_state: Random seed for reproducibility.
        :return: Trained Random Forest model and evaluation metrics.
        """
        # Train the model
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Random Forest Regression Results:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        # SHAP analysis
        print("Calculating SHAP values...")
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        # Debug: Check SHAP values
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Example SHAP values: {shap_values[:5]}")

        # Debug: Check feature importance
        feature_importances = rf.feature_importances_
        print("Top 10 Feature Importances:")
        for feature, importance in sorted(zip(X_test.columns, feature_importances), key=lambda x: x[1], reverse=True)[:25]:
            print(f"{feature}: {importance:.4f}")

        # Debug: Check for variability in SHAP values
        shap_variability = shap_values.var(axis=0)
        print(f"SHAP variability: {shap_variability}")

        # Display SHAP beeswarm plot
        print("Displaying SHAP beeswarm plot...")
        shap.summary_plot(shap_values, X_test, max_display=40, plot_type="dot")

        return rf, {"MAE": mae, "MSE": mse, "R2": r2}

    def random_forest_pipeline_with_shap(self, data, target_column="tothlos", test_size=0.2, random_state=42):
        """
        Full pipeline to train and evaluate a Random Forest regression model,
        with SHAP value visualization.

        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :param test_size: Proportion of the data to use as the test set.
        :param random_state: Random seed for reproducibility.
        :return: Trained Random Forest model and evaluation metrics.
        """
        print(f"Preparing pipeline for target: {target_column}")

        # Adjust for binary targets
        data = self.adjust_for_binary_targets(data, target_column)

        # Split the data
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = self.prepare_train_test(data, target_column, test_size, random_state)

        # Train and evaluate the model with SHAP
        print("Training Random Forest model with SHAP visualization...")
        return self.run_random_forest_regression_with_shap(X_train, X_test, y_train, y_test)
    
    def load_and_clean_data(self):
        """
        Loads and cleans the scoliosis data from the specified CSV path.
        :return: Cleaned DataFrame.
        """
        try:
            # Load data
            data = pd.read_csv(self.csv_path)

            # Basic cleaning
            data.drop_duplicates(inplace=True)
            data.fillna(method="ffill", inplace=True)
            data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")

            # Convert date columns to datetime
            for date_col in ["date_of_surgery", "date_of_birth"]:
                if date_col in data.columns:
                    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")

            return data
        except FileNotFoundError:
            print(f"Error: File not found at {self.csv_path}")
            return pd.DataFrame()
    
    def run_random_forest_regression(self, X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
        """
        Fits a Random Forest Regressor and evaluates it on the test set.
        
        :param X_train: Training features.
        :param X_test: Testing features.
        :param y_train: Training target.
        :param y_test: Testing target.
        :param n_estimators: Number of trees in the Random Forest.
        :param random_state: Random seed for reproducibility.
        :return: Trained Random Forest model and evaluation metrics.
        """
        # Train the model
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        rf.fit(X_train, y_train)

        # Make predictions
        y_pred = rf.predict(X_test)

        # Evaluate the model
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Random Forest Regression Results:")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Mean Squared Error (MSE): {mse:.2f}")
        print(f"R² Score: {r2:.2f}")

        return rf, {"MAE": mae, "MSE": mse, "R2": r2}
    
    def adjust_for_binary_targets(self, data, target_column):
        """
        Adjusts the target column for binary targets (if detected).
        
        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :return: Adjusted DataFrame with binary target encoded as 0/1 if necessary.
        """
        if data[target_column].nunique() == 2:
            print(f"Binary target detected for '{target_column}'. Encoding as 0/1.")
            data[target_column] = data[target_column].astype(int)
        return data
    
    def random_forest_pipeline_with_binary(self, data, target_column="tothlos", test_size=0.2, random_state=42):
        """
        Full pipeline to train and evaluate a Random Forest regression model,
        with adjustments for binary targets.
        
        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :param test_size: Proportion of the data to use as the test set.
        :param random_state: Random seed for reproducibility.
        :return: Trained Random Forest model and evaluation metrics.
        """
        print(f"Preparing pipeline for target: {target_column}")
        
        # Adjust for binary targets
        data = self.adjust_for_binary_targets(data, target_column)
        
        # Run the pipeline
        return self.random_forest_pipeline(data, target_column, test_size, random_state)


    def random_forest_pipeline(self, data, target_column="tothlos", test_size=0.2, random_state=42):
        """
        Full pipeline to train and evaluate a Random Forest regression model on a given dataset.
        
        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :param test_size: Proportion of the data to use as the test set.
        :param random_state: Random seed for reproducibility.
        :return: Trained Random Forest model and evaluation metrics.
        """
        print(f"Running Random Forest Regression for target: {target_column}")
        
        # Split the data
        X_train, X_test, y_train, y_test = self.prepare_train_test(data, target_column, test_size, random_state)
        
        # Train and evaluate the model
        rf_model, metrics = self.run_random_forest_regression(X_train, X_test, y_train, y_test, random_state=random_state)
        
        return rf_model, metrics

    # Updated Method
    def generate_training_dataframe(self, target_col="any_ssi"):
        
        
        data = self.load_and_clean_data()
        if data.empty:
            return pd.DataFrame()

        # Lowercase columns
        data.columns = data.columns.str.lower()

        # 1 check if target column present
        # 2 if not present check known_composites
        # 3 if in known composites, generate target from composite columns and remove those from dataframe once target column is established.

        data = ScoliosisFeatureEngineeringService.generate_comp_target(data, target_col=target_col)
        data = ScoliosisFeatureEngineeringService.drop_perfectly_correlated_columns(data, target_column=target_col)

        def replace_with_nan(data, missing_tokens=None):
            """
            Replaces a set of tokens with np.nan across all columns in 'data',
            using case-insensitive regex matching. Also trims whitespace so that
            e.g. "  -99 " becomes NaN if "-99" is in missing_tokens.

            :param data: Pandas DataFrame
            :param missing_tokens: list of tokens to treat as missing
            :return: DataFrame with those tokens replaced by NaN
            """
            if missing_tokens is None:
                missing_tokens = ["-99", "null", "#null!", "na", "n/a", "", " "]

            # 1) Convert every column to string, strip leading/trailing whitespace
            for col in data.columns:
                data[col] = data[col].astype(str).str.strip()

            # 2) For each token, do a regex-based replace (case-insensitive)
            for token in missing_tokens:
                # Example pattern:   ^(?i)\s*null\s*$
                #  - (?i) case-insensitive
                #  - ^...$ match entire string
                # We already stripped, so we don’t specifically need \s*, but you can add it if you want
                pattern = rf"(?i)^{re.escape(token)}$"
                data.replace(to_replace=pattern, value=np.nan, regex=True, inplace=True)
                
            return data
        
        
        columns_to_drop_if_empty = ["ped_sap_name1"]
        data = ScoliosisFeatureEngineeringService.remove_empty_rows(data, columns_to_drop_if_empty)

        # 1) Replace placeholders with NaN
        data = replace_with_nan(
            data,
            missing_tokens=["-99", "null", "#null!", "na", "n/a", "", " "]
        )

        # 2) Encode binary 
        data = ScoliosisFeatureEngineeringService.encode_binary(data, POTENTIAL_BINARY)
        # 3) Identify & encode categorical
        cat_cols = ScoliosisFeatureEngineeringService.find_potential_string_categorical_cols(data)
        data = ScoliosisFeatureEngineeringService.encode_categorical(data, cat_cols)

            # --- Now handle antibiotic regimen logic ---
        # Step 1: Collapse the ped_sap_nameN columns
        data = ScoliosisFeatureEngineeringService.multi_hot_encode_abx(data)
        data = ScoliosisFeatureEngineeringService.encode_abx_combinations(data)
        data = ScoliosisFeatureEngineeringService.encode_bmi(data)
        # data = ScoliosisFeatureEngineeringService.collapse_abx_columns(data)
        
        # # Step 2: Derive a single label 'abx_regimen'
        # data = ScoliosisFeatureEngineeringService.categorize_abx_regimen(data)

        # Step 3: (Optional) pivot to multiple True/False columns for each regimen
        # data = ScoliosisFeatureEngineeringService.pivot_regimen_flags(data)
        
        # data = ScoliosisFeatureEngineeringService.encode_categorical(data, ["abx_regimen"])



        # 4) Add derived features, etc.
        
        # data = ScoliosisFeatureEngineeringService.add_derived_features(data)

        # 5) Possibly drop or keep some columns if needed
        data = ScoliosisFeatureEngineeringService.drop_low_variance_columns(data)

        # 6) Correlation
        

        # 7) Clip numeric range
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].clip(
            np.finfo(np.float32).min,
            np.finfo(np.float32).max
        )

        # 8) Make sure columns are numeric-friendly
        data = ScoliosisFeatureEngineeringService.make_rf_compatible(data)

        # 9) Rename columns for XGBoost if needed
        data = ScoliosisFeatureEngineeringService.rename_for_xgb_compatibility(data)

        return data
        
    
    def get_df(self, force_load=False):
        """
        Retrieves or generates the training DataFrame for scoliosis predictions.
        :param force_load: Forces regeneration of the DataFrame if True.
        :return: Training DataFrame.
        """
        if self.aggregated_stats and not force_load:
            return pickle.loads(self.aggregated_stats)

        df = self.generate_training_dataframe()
        if df.empty:
            return None

        # Save aggregated stats to the database
        self.aggregated_stats = memoryview(pickle.dumps(df))
        self.update(aggregated_stats=self.aggregated_stats)
        return df

    def save(self):
        """
        Saves the current ScoliosisTimePredictor instance to the SQLite database.
        """
        self.updated_at = datetime.now().isoformat()
        super().save()

    @classmethod
    def load(cls, predictor_id):
        """
        Loads a ScoliosisTimePredictor instance by ID from the database.
        :param predictor_id: ID of the predictor to load.
        :return: ScoliosisTimePredictor instance or None if not found.
        """
        row = cls._cursor.execute(f"SELECT * FROM {cls.table_name} WHERE id=?", (predictor_id,)).fetchone()

        if row:
            return cls(
                id=row["id"],
                task_id=row["task_id"],
                metrics=row["metrics"],
                matrix=row["matrix"],
                csv_path=row["csv_path"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                cohort_id=row["cohort_id"],
                aggregated_stats=row["aggregated_stats"],
                aggregated_stats_non_normed=row["aggregated_stats_non_normed"],
            )
        else:
            print(f"No ScoliosisTimePredictor found with ID {predictor_id}.")
            return None



class ScoliosisFeatureEngineeringService:

    @staticmethod
    def generate_comp_target(data, target_col, axis=1):
        def standardize_ssi_column(value):
            """
            Convert an SSI-like string/number to 0 or 1.
            For example, 'No Complication', '-99', '' => 0.
            Everything else => 1.
            You can further refine logic if your data has different placeholders.
            """
            val_str = str(value).strip().lower()
            # For example, treat these as "no infection"
            if val_str in ["no complication", "-99", "nan", "", "null", "none"]:
                return 0
            else:
                # If it's anything else, we treat as 1
                return 1
        # If the requested target_col is in KNOWN_COMPOSITES,
        # but not already present in data, let's build it.
        if target_col not in data.columns and target_col in KNOWN_COMPOSITES:
            composite_cols = KNOWN_COMPOSITES[target_col]

            # Check for missing columns
            missing = [c for c in composite_cols if c not in data.columns]
            if missing:
                print(f"Warning: Composite target '{target_col}' missing columns: {missing}.")

            # Initialize the new target column as all zeros
            data[target_col] = 0

            # For each column that does exist, standardize => 0/1, then OR it in
            for c in composite_cols:
                if c in data.columns:
                    # 1) Convert to 0 or 1
                    data[c] = data[c].apply(standardize_ssi_column).astype(int)
                    # 2) Combine into the composite
                    data[target_col] = data[target_col] | data[c]

            print(f"Created composite column '{target_col}' from {composite_cols}.")

        # If after that, the target_col is still not in data, we can't proceed
        if target_col not in data.columns:
            print(f"Target column '{target_col}' not found in data. Returning empty DataFrame.")
            return pd.DataFrame()

        return data
    
    @staticmethod
    def multi_hot_encode_abx(df: pd.DataFrame) -> pd.DataFrame:
        """
        (Earlier method) Create multi-hot (binary) columns for each antibiotic 
        from columns starting with "ped_sap_name". This method should have been 
        run first so that the DataFrame has columns like used_cefazolin, used_vancomycin, etc.
        """
        import re
        # Map messy names to canonical names
        ABX_NAME_MAP = {
            'ampicillin without sulbactam': 'ampicillin',
            'ampicillin':                  'ampicillin',
            'cefazolin':                   'cefazolin',
            'cefoxitin':                   'cefoxitin',
            'ceftazidime':                 'ceftazidime',
            'ceftriaxone':                 'ceftriaxone',
            'vancomycin':                  'vancomycin',
            'gentamicin':                  'gentamicin',
            'clindamycin':                 'clindamycin',
            'metronidazole':               'metronidazole',
            'doxycycline':                 'doxycycline',
        }
        
        # Initialize the "used_<abx>" columns to False
        unique_abx = set(ABX_NAME_MAP.values())
        for abx in unique_abx:
            df[f"used_{abx}"] = False
        
        # Loop over all columns that start with "ped_sap_name" to update the used_* flags.
        for col in df.columns:
            if not col.lower().startswith("ped_sap_name"):
                continue
            # Remove the prefix to get the antibiotic name part and lowercase it.
            raw_abx_name = re.sub(r"^ped_sap_name\d+_", "", col, flags=re.IGNORECASE).lower()
            for pattern, canonical in ABX_NAME_MAP.items():
                if pattern in raw_abx_name:
                    df.loc[df[col] == True, f"used_{canonical}"] = True
                    break
        return df

    @staticmethod
    def encode_abx_combinations(df: pd.DataFrame) -> pd.DataFrame:
        """
        Iterates row by row over the multi-hot encoded antibiotic columns 
        (columns starting with "used_") and determines the unique combination of 
        antibiotics used. For each unique combination encountered, a new column is 
        added to the DataFrame. The new column is named based on the antibiotics 
        present (e.g., "combo_ceftriaxone_vancomycin"). For a row that has that 
        combination, that column is set to 1; all other rows will have 0 in that column.
        """
        # Get all columns that begin with "used_"
        abx_cols = [col for col in df.columns if col.startswith("used_")]
        
        # Dictionary to keep track of combination (as a tuple) -> new column name
        combo_map = {}
        
        # Iterate row by row
        for idx, row in df.iterrows():
            # Build a list of antibiotics used in this row.
            # Remove the "used_" prefix to get a cleaner name.
            used_list = [col.replace("used_", "") for col in abx_cols if row[col] == True or row[col] == 1]
            used_list.sort()  # sort for consistency
            combo_key = tuple(used_list)
            
            # Create a column name. If no antibiotic was used, label it as "combo_none".
            if combo_key:
                # Join the antibiotic names with underscores.
                combo_col = "combo_abx_regimen_" + "_".join(combo_key)
            else:
                combo_col = "single_abx_regimen_"
            
            # If this combination has not been seen, add it as a new column (initialize with 0)
            if combo_key not in combo_map:
                combo_map[combo_key] = combo_col
                # Add the new column with 0 for all rows.
                df[combo_col] = 0
            
            # For this row, mark the new combination column as 1.
            df.at[idx, combo_map[combo_key]] = 1
        
        return df
    ############################################################
    # 1) Collapse ped_sap_name columns -> used_<antibiotic> cols
    ############################################################
    @staticmethod
    def collapse_abx_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a single True/False column 'used_<abx>' for each antibiotic 
        by scanning all 'ped_sap_nameN_' columns in df.
        """

        # Map the messy column name substring -> a canonical antibiotic key
        ABX_NAME_MAP = {
            'ampicillin without sulbactam': 'ampicillin',
            'ampicillin':                  'ampicillin',
            'cefazolin':                   'cefazolin',
            'cefoxitin':                   'cefoxitin',
            'ceftazidime':                 'ceftazidime',
            'ceftriaxone':                 'ceftriaxone',
            'vancomycin':                  'vancomycin',
            'gentamicin':                  'gentamicin',
            'clindamycin':                 'clindamycin',
            'metronidazole':               'metronidazole',
            'doxycycline':                 'doxycycline',
            # Add more patterns/antibiotics here if needed
        }

        # Initialize “used_xxx” columns as False
        unique_abx = set(ABX_NAME_MAP.values())  # e.g. {'ampicillin','cefazolin','vancomycin',...}
        for abx_key in unique_abx:
            df[f"used_{abx_key}"] = False

        # For each ped_sap_nameN_ column, set the relevant used_xxx = True where that row is True
        for col in df.columns:
            if not col.startswith("ped_sap_name"):
                continue

            # Extract the antibiotic name portion from the column name
            # e.g. "ped_sap_name1_CeFAZoLin (Kefzol)" -> "cefazolin (kefzol)"
            raw_abx_name = re.sub(r"^ped_sap_name\d+_", "", col, flags=re.IGNORECASE).lower()

            # Attempt to find which antibiotic pattern matches from ABX_NAME_MAP
            for pattern, canonical_abx in ABX_NAME_MAP.items():
                if pattern in raw_abx_name:
                    # Rows where this column == True => used_<canonical_abx> = True
                    df.loc[df[col] == True, f"used_{canonical_abx}"] = True
                    break

        return df

    #####################################################
    # 2) Derive single-column antibiotic regimen labels
    #####################################################
    @staticmethod
    def categorize_abx_regimen(df: pd.DataFrame, axis=1) -> pd.DataFrame:
        """
        Looks at the 'used_<abx>' columns and assigns a single 
        classification label to each row in a new column 'abx_regimen'.
        """

        # Which antibiotics will we check for?
        # You can keep these in sync with ABX_NAME_MAP above or 
        # just define them explicitly if you only care about a subset.
        ALL_ABX = [
            "ampicillin", 
            "cefazolin", "cefoxitin", "ceftazidime", "ceftriaxone",
            "vancomycin", "gentamicin", "clindamycin", 
            "metronidazole", "doxycycline"
            # etc.
        ]

        # Define your regimen rules (exact matches) in priority order
        REGIMEN_RULES = [
            # Single-agent
            ({"cefazolin"},            "cef_alone"),
            ({"vancomycin"},           "vanco_alone"),
            ({"gentamicin"},           "gent_alone"),
            ({"clindamycin"},          "clinda_alone"),
            ({"ceftriaxone"},          "ceftriaxone_alone"),
            ({"ampicillin"},           "ampicillin_alone"),

            # Two-agent combos
            ({"cefazolin", "gentamicin"},   "cef_gent"),
            ({"vancomycin", "clindamycin"}, "vanco_clinda"),
            ({"cefazolin", "vancomycin"},   "cef_vanco"),
            ({"ceftriaxone", "vancomycin"}, "ceftriaxone_vanco"),
            ({"cefazolin", "clindamycin"},  "cef_clinda"),
            # Add more combos as needed...
        ]

        # Helper function to look at each row, gather used ABX, match a rule
        def assign_regimen(row):
            # gather set of used antibiotics
            used = set(abx for abx in ALL_ABX if row.get(f"used_{abx}", False))

            # Attempt to match exact sets first
            for required_set, label in REGIMEN_RULES:
                if used == required_set:
                    return label

            # Fallback logic
            if len(used) >= 3:
                return "multi_abx"
            elif len(used) == 0:
                return "none_used"
            else:
                return "other"

        df["abx_regimen"] = df.apply(assign_regimen, axis=axis)
        return df

    ##########################################################
    # 3) Pivot final regimen labels into multiple columns
    ##########################################################
    @staticmethod
    def pivot_regimen_flags(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts the single 'abx_regimen' column into multiple 
        True/False columns (one for each possible regimen label).
        """
        # Define which labels we want to pivot
        # (includes fallback labels: multi_abx, other, none_used)
        possible_regimens = [
            "cef_alone", "vanco_alone", "gent_alone", "clinda_alone", 
            "ceftriaxone_alone", "ampicillin_alone",
            "cef_gent", "vanco_clinda", "cef_vanco", "ceftriaxone_vanco", 
            "cef_clinda", 
            "multi_abx", "other", "none_used"
        ]

        for regimen_label in possible_regimens:
            col_name = f"abx_regimen_{regimen_label}"
            df[col_name] = (df["abx_regimen"] == regimen_label)
        return df

    
    @staticmethod
    def encode_binary(data, binary_columns):
        """
        For each column in 'binary_columns', tries to encode it as 0/1 using a textual map
        plus a numeric check. If the column has any numeric values outside {0,1,-99,-1},
        or textual tokens not in the dictionary, we skip encoding for that column entirely.
        
        :param data: pandas DataFrame
        :param binary_columns: list of column names suspected to be binary
        :return: DataFrame with those columns conditionally encoded as 0/1
        """
        
        # Textual synonyms mapped to 0 or 1
        binary_map = {
            # TRUE synonyms
            "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1, "male": 1, "Yes": 1,

            # FALSE synonyms
            "no": 0, "n": 0, "false": 0, "f": 0, "0": 0, "female": 0, "No": 0,

            # # Values you'd like to force to 0 instead of missing
            # "null": NaN, "#null!": 0, "none": 0, "na": 0, "n/a": 0, "-99": 0
        }

        # Numeric values that we allow if they're present (rest => skip the column)
        allowed_numerics = {0.0, 1.0, -99.0, -1.0}

        for col in binary_columns:
            col_lower = col.lower()
            if col_lower not in data.columns:
                continue  # skip if it doesn't exist

            # Save original data in case we revert
            original_series = data[col_lower].copy()

            # Gather all distinct non‐NaN values
            distinct_vals = original_series.dropna().unique()

            # Flag to see if we skip encoding
            skip_encoding = False

            # 1) Quick scan of all distinct values
            for val in distinct_vals:
                str_val = str(val).strip().lower()

                if str_val in binary_map:
                    # e.g. 'yes','no','male','null'
                    continue
                else:
                    # If not in dictionary, try numeric parse
                    try:
                        float_val = float(str_val)
                        # If the parsed float is not in allowed_numerics => skip
                        if float_val not in allowed_numerics:
                            skip_encoding = True
                            break
                    except ValueError:
                        # Not parseable as float => skip
                        print("Not parse", str_val)
                        skip_encoding = True
                        break

            if skip_encoding:
                # revert to original
                data[col_lower] = original_series
                continue

            # 2) Actually encode the column
            def map_to_binary(x):
                """Helper function to map x => 0/1 or NaN."""
                s = str(x).strip().lower()
                if s in binary_map:
                    return float(binary_map[s])  # 0.0 or 1.0
                else:
                    # Try numeric parse for 0/1/-99/-1
                    try:
                        f_val = float(s)
                        if f_val in allowed_numerics:
                            # You might want to treat -99, -1 as 0 or NaN
                            # By default let's do 0 here
                            return 0.0 if f_val != 1.0 else 1.0
                        else:
                            return np.nan
                    except ValueError:
                        return np.nan

            temp_encoded = original_series.apply(map_to_binary)
            unique_after = temp_encoded.dropna().unique()

            # 3) Ensure truly binary
            if len(unique_after) <= 2:
                data[col_lower] = temp_encoded
            else:
                # Not truly binary, revert
                data[col_lower] = original_series

        return data

    @staticmethod
    def find_potential_string_categorical_cols(df, max_unique=50):
        """
        Identifies columns in 'df' whose dtype is object/category/string,
        that have fewer than 'max_unique' distinct values,
        and that are not purely boolean-like or empty.
        """
        bool_like_values = {
            "yes", "no", "true", "false", "0", "1", "null",
            "none", "", "nan", "-99", "-1", "male", "female"
        }

        text_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        potential_categorical = []
        
        for col in text_cols:
            unique_vals = df[col].unique()  # includes NaN
            unique_non_nan = [str(v).strip() for v in unique_vals if pd.notna(v)]

            if len(unique_vals) < max_unique:
                normalized = set(val.lower() for val in unique_non_nan)
                # If they're all in the bool-like set, skip it
                if not normalized.issubset(bool_like_values):
                    potential_categorical.append(col)

        return potential_categorical
    
    @staticmethod
    def rename_for_xgb_compatibility(df):
        """
        Renames columns to remove XGBoost-forbidden characters (like [ ] < > ( )) 
        and ensures all column names are strings. Returns the modified DataFrame.
        """
        # Convert all column names to string, just in case
        df.columns = df.columns.map(str)

        # Replace each forbidden character with underscore (or remove them)
        forbidden_pattern = r"[\[\]<>\(\)]"
        df.columns = [re.sub(forbidden_pattern, "_", col) for col in df.columns]

        return df

    @staticmethod
    def encode_categorical(data, categorical_columns, max_categories=50, numeric_threshold=0.9):
        """
        One-hot encodes categorical columns with limited unique values.
        Groups rare categories into "Other" if necessary.
        Skips columns where most values (defined by numeric_threshold) are numbers.
        
        :param data: pandas DataFrame
        :param categorical_columns: list of column names to potentially encode
        :param max_categories: maximum number of unique categories before grouping
        :param numeric_threshold: if proportion of numeric values exceeds this, skip encoding (default 0.9)
        :return: DataFrame with categorical columns encoded
        """
        for col in categorical_columns:
            col_lower = col.lower()
            if col_lower in data.columns:
                # Get non-null values
                non_null_values = data[col_lower].dropna()
                
                try:
                    # Convert to numeric and check proportion of successful conversions
                    numeric_values = pd.to_numeric(non_null_values, errors='coerce')
                    numeric_proportion = numeric_values.notna().mean()
                    
                    # Skip if proportion of numeric values exceeds threshold
                    if numeric_proportion >= numeric_threshold:
                        continue
                    
                    # Proceed with encoding for non-numeric columns
                    unique_values = data[col_lower].nunique()
                    if unique_values > max_categories:
                        # Group rare categories into "Other"
                        top_categories = data[col_lower].value_counts().nlargest(max_categories).index
                        data[col_lower] = data[col_lower].apply(lambda x: x if x in top_categories else "Other")
                    
                    # Apply one-hot encoding
                    data = pd.get_dummies(data, columns=[col_lower], drop_first=True)
                
                except (TypeError, ValueError):
                    # If conversion to numeric fails entirely, treat as categorical
                    unique_values = data[col_lower].nunique()
                    if unique_values > max_categories:
                        top_categories = data[col_lower].value_counts().nlargest(max_categories).index
                        data[col_lower] = data[col_lower].apply(lambda x: x if x in top_categories else "Other")
                    data = pd.get_dummies(data, columns=[col_lower], drop_first=True)
        
        return data

    @staticmethod
    def handle_missing(data):
        """
        Replaces placeholders like '#NULL!' and -99 with NaN and drops entirely NaN columns.
        """
        data.replace({"#NULL!": np.nan, -99: np.nan, "-99.00": np.nan}, inplace=True)
        data.dropna(axis=1, how="all", inplace=True)
        return data
    
    @staticmethod
    def encode_true_false_columns(data):
        """
        Identifies columns with only True/False values and encodes them as binary (0/1).
        """
        for col in data.columns:
            if data[col].dtype == 'bool':  # Check if the column is boolean
                data[col] = data[col].astype(int)  # Convert True/False to 1/0
        return data

    @staticmethod
    def add_derived_features(data):
        """
        Adds derived features like age in years if age in days is present.
        """
        if "age_days" in data.columns:
            data["age_years"] = data["age_days"] / 365
        return data
    
    @staticmethod
    def make_rf_compatible(data):
        """
        Processes the DataFrame to handle missing values and ensure compatibility with RandomForest models.
        """
        # Step 1: Identify columns with missing values
        missing_cols = data.columns[data.isna().any()].tolist()
        print(f"Columns with missing values: {missing_cols}")

        # Step 2: Handle missing values based on column types
        for col in missing_cols:
            if data[col].dtype in [np.float64, np.int64]:  # Numeric columns
                # Fill numeric columns with the mean
                data[col].fillna(data[col].mean(), inplace=True)
            elif data[col].dtype == "object" or data[col].dtype.name == "category":  # Categorical columns
                # Fill categorical columns with a placeholder "Unknown"
                data[col].fillna("Unknown", inplace=True)
                # Convert to strings to ensure uniformity
                data[col] = data[col].astype(str)
                # Convert categorical columns to numeric using LabelEncoder
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
            elif data[col].dtype == "bool":  # Boolean columns
                # Fill boolean columns with 0 (False)
                data[col].fillna(False, inplace=True)
                data[col] = data[col].astype(int)
            else:
                # Default fallback for other types
                data[col].fillna(0, inplace=True)

        # Step 3: Convert all non-numeric columns to numeric
        for col in data.select_dtypes(include=["object", "category"]).columns:
            # Convert to strings, then apply LabelEncoder
            data[col] = data[col].astype(str)
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])


        # Step 4: Verify no missing values remain
        if data.isna().any().any():
            print("Warning: NaN values remain after processing.")
        else:
            print("All missing values handled successfully.")

        # Check for mixed types

        return data
    
    @staticmethod
    def remove_empty_rows(data, columns_to_check):
        """
        Removes rows that have an empty string (or only whitespace) or NaN in ANY of the specified columns.
        If a row has at least one column in 'columns_to_check' where the value is '', NaN, or whitespace, that row is removed.
        
        :param data: pandas DataFrame
        :param columns_to_check: list of column names to check for empties
        :return: DataFrame with those rows removed
        """
        # Start with a mask of True for all rows
        mask = pd.Series(True, index=data.index)
        
        for col in columns_to_check:
            if col in data.columns:
                # First check for NaN values using pd.isna()
                not_nan = ~pd.isna(data[col])
                # For non-NaN values, check if they're empty strings
                # Only convert to string and check emptiness for non-NaN values
                not_empty = pd.Series(True, index=data.index)  # Default to True
                non_nan_mask = ~pd.isna(data[col])
                if non_nan_mask.any():
                    not_empty[non_nan_mask] = data.loc[non_nan_mask, col].astype(str).str.strip() != ''
                
                # Combine the conditions: keep rows that are neither NaN nor empty
                mask = mask & (not_nan & not_empty)
            else:
                print(f"Warning: Column '{col}' does not exist in DataFrame. Skipping.")
        
        # Filter the DataFrame to only rows that are True in 'mask'
        return data[mask]
    
    @staticmethod
    def remove_highly_correlated_features(data, threshold=0.9):
        """
        Removes highly correlated features from the DataFrame.
        
        :param data: Input DataFrame.
        :param threshold: Correlation threshold for removing features.
        :return: DataFrame with redundant features removed.
        """
        print("Removing highly correlated features...")

        # Compute correlation matrix
        corr_matrix = data.corr().abs()

        # Upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation greater than the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        print(f"Highly correlated features to drop: {to_drop}")

        # Drop the redundant features
        data = data.drop(columns=to_drop)

        return data
    
    @staticmethod
    def make_unique_column_names(columns):
        """
        Given a list of column names, rename duplicates by appending .1, .2, etc.
        Example: If "COL" appears 3 times, they become "COL", "COL.1", "COL.2".
        Returns a new list of unique column names in the same order.
        """
        new_cols = []
        name_count = {}

        for col in columns:
            if col not in name_count:
                # first occurrence
                name_count[col] = 0
                new_cols.append(col)
            else:
                # duplicate: increment and rename
                name_count[col] += 1
                new_cols.append(f"{col}.{name_count[col]}")

        return new_cols
    
    @staticmethod
    def encode_bmi(data):
        """
        Enhanced BMI calculation with robust type checking and debugging
        """
        print("\n=== BMI Encoding Process ===")
        
        # Check for required columns
        required_cols = {'height', 'weight'}
        missing_cols = required_cols - set(data.columns)
        if missing_cols:
            print(f"🚨 Critical: Missing required columns {missing_cols} - skipping BMI calculation")
            return data

        # Create numeric copies with error checking
        def safe_convert(col, conversion_factor):
            """Convert column to numeric with detailed error reporting"""
            try:
                # First attempt numeric conversion
                series = pd.to_numeric(data[col], errors='coerce')
                
                # Log conversion success rate
                non_convertible = series.isna() & data[col].notna()
                if non_convertible.any():
                    invalid_values = data.loc[non_convertible, col].unique()[:5]
                    print(f"  ⚠️ Non-numeric values in {col}: {len(invalid_values)} unique (e.g., {invalid_values})")
                
                # Apply unit conversion
                return series * conversion_factor
            except Exception as e:
                print(f"🚨 Critical error converting {col}: {str(e)}")
                return pd.Series(np.nan, index=data.index)

        print("\n1. Unit Conversion:")
        print("  • Converting height (inches → meters)")
        df = data.assign(
            height_m=safe_convert('height', 0.0254),
            weight_kg=safe_convert('weight', 0.453592)
        )
        
        # # Conversion metrics
        # print(f"  • Height conversion summary:")
        # print(f"    - Original units: inches (range {df.height.min():.1f}-{df.height.max():.1f})")
        # print(f"    - Converted to meters (range {df.height_m.min():.2f}-{df.height_m.max():.2f})")
        
        # print(f"\n  • Weight conversion summary:")
        # print(f"    - Original units: pounds (range {df.weight.min():.1f}-{df.weight.max():.1f})")
        # print(f"    - Converted to kg (range {df.weight_kg.min():.1f}-{df.weight_kg.max():.1f})")

        # --------------------------------------------------
        # 2. Data Validation
        # --------------------------------------------------
        print("\n2. Data Validation:")
        
        # Physiological plausibility checks
        height_valid = df['height_m'].between(0.3, 2.5)  # ~12-100 inches
        weight_valid = df['weight_kg'].between(2, 300)   # ~4.4-660 lbs
        
        df['height_m'] = np.where(height_valid, df['height_m'], np.nan)
        df['weight_kg'] = np.where(weight_valid, df['weight_kg'], np.nan)

        # Log validation results
        print(f"  • Height validation:")
        print(f"    - Valid range: 0.3-2.5 meters (12-100 inches)")
        print(f"    - Invalid values: {height_valid.sum():,}/{len(df)} ({height_valid.mean()*100:.1f}%) valid")
        
        print(f"\n  • Weight validation:")
        print(f"    - Valid range: 2-300 kg (4.4-660 lbs)")
        print(f"    - Invalid values: {weight_valid.sum():,}/{len(df)} ({weight_valid.mean()*100:.1f}%) valid")

        # --------------------------------------------------
        # 3. BMI Calculation
        # --------------------------------------------------
        print("\n3. BMI Calculation:")
        
        # Calculate BMI with zero division protection
        with np.errstate(divide='ignore', invalid='ignore'):
            df['bmi'] = df['weight_kg'] / (df['height_m'] ** 2)

        valid_bmi = df['bmi'].notna()
        print(f"  • Successful calculations: {valid_bmi.sum():,}/{len(df)}")
        print(f"  • BMI range: {df.bmi.min():.1f}-{df.bmi.max():.1f}")

        # --------------------------------------------------
        # 4. Quality Flagging
        # --------------------------------------------------
        print("\n4. Quality Flags:")
        
        # Clinical plausibility thresholds (WHO standards)
        bmi_conditions = [
            (df['bmi'] < 10) | (df['bmi'] > 60),    # Extreme values
            (df['bmi'] < 15) | (df['bmi'] > 40),    # Clinically improbable
            (df['bmi'] < 18.5) | (df['bmi'] > 30)   # Unusual but possible
        ]
        
        bmi_flags = [
            'invalid_extreme',
            'invalid_clinical', 
            'unusual_clinical'
        ]
        
        df['bmi_quality_flag'] = np.select(
            condlist=bmi_conditions,
            choicelist=bmi_flags,
            default='valid'
        )

        # Flag distribution statistics
        flag_counts = df['bmi_quality_flag'].value_counts()
        print("  • BMI quality distribution:")
        for flag, count in flag_counts.items():
            print(f"    - {flag}: {count:,} ({count/len(df)*100:.1f}%)")

        # --------------------------------------------------
        # 5. Final Integration
        # --------------------------------------------------
        print("\n5. Final Dataset Integration:")
        
        # Clean implausible BMIs while preserving original data
        data['bmi'] = np.where(
            df['bmi_quality_flag'].isin(['valid', 'unusual_clinical']),
            df['bmi'],
            np.nan
        )
        
        data['bmi_quality_flag'] = df['bmi_quality_flag']
        
        final_valid = data['bmi'].notna().sum()
        # print(f"  • Final valid BMI values: {final_valid:,}/{initial_count:,}")
        print(f"  • Null BMI values: {data['bmi'].isna().sum():,}")
        print("✅ BMI encoding complete\n")
        
        return data



    @staticmethod
    def drop_low_variance_columns(data, threshold=0.001):
        """
        Drops columns with very low variance or only one unique value.
        Also ensures columns are uniquely named so data[col] is always a Series.

        :param data: Input DataFrame
        :param threshold: Variance threshold below which columns are dropped
        :return: DataFrame with low-variance columns removed
        """
        print("Checking for low-variance columns...")

        # 0) Ensure column names are unique. If not, rename them.
        if data.columns.duplicated().any():
            print("Warning: Duplicate column names found; renaming them.")
            data.columns = ScoliosisFeatureEngineeringService.make_unique_column_names(data.columns)

        # 1) Identify columns that have exactly one unique value (including NaN)
        single_unique_cols = []
        for col in data.columns:
            # Now data[col] should be a Series, not a DataFrame
            if data[col].nunique(dropna=False) == 1:
                single_unique_cols.append(col)

        # 2) Identify numeric columns with variance < threshold
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        low_var_numeric_cols = []
        for col in numeric_cols:
            # Convert forcibly to numeric, ignoring errors
            col_series = pd.to_numeric(data[col], errors='coerce').dropna()
            if col_series.empty:
                # Entire column was NaNs or not numeric => treat variance as 0
                col_var = 0.0
            else:
                col_var = col_series.var()
            
            if pd.isna(col_var):
                col_var = 0.0
            
            if col_var < threshold:
                low_var_numeric_cols.append(col)

        # 3) Combine both sets
        low_variance_cols = list(set(single_unique_cols + low_var_numeric_cols))
        print(f"Low-variance columns to drop: {low_variance_cols}")

        # 4) Drop those columns
        data = data.drop(columns=low_variance_cols, errors="ignore")
        data = data.drop(columns="htooday", errors="ignore")
        return data

    
    @staticmethod
    def drop_perfectly_correlated_columns(data, target_column):
        """
        Drops columns that are perfectly or highly correlated with the target column.

        :param data: Input DataFrame.
        :param target_column: The column to predict.
        :return: DataFrame with correlated columns dropped.
        """
        print(f"Checking for columns perfectly correlated with {target_column}...")

        # Select only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])

        if target_column not in numeric_data.columns:
            raise ValueError(f"Target column '{target_column}' must be numeric for correlation analysis.")

        # Compute correlation matrix for numeric columns
        corr_matrix = numeric_data.corr()
        
        # Get columns perfectly correlated with the target (correlation = 1)

        correlated_columns = corr_matrix.index[corr_matrix[target_column] > .8].tolist()
        # correlated_columns = ['doptodis','tothlos','htooday']
        correlated_columns = ["dsupinfec", "doptodis", "wndinfd", "orgspcssi", "dehis", "nwnd", "dwndinfd", "dorgspcssi", "ddehis", "ndehis", "norgspcssi", "nwndinfd", "noupneumo", "doupneumo", "reopor2cpt1", "retor2related", "unplannedreadmission1", "reoporcpt1"]
        
        # correlated_columns.remove(target_column)  # Exclude the target itself
        
        # TODO: Remove hospital_stay day stuff then get rid of this line.
        

        print(f"Columns perfectly correlated with {target_column}: {correlated_columns}")

        # Drop correlated columns from the original DataFrame
        data = data.drop(columns=correlated_columns, errors="ignore")
        return data