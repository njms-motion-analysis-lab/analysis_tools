
import pickle
import pandas as pd
from datetime import datetime
from prediction_tools.time_predictor import TimePredictor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import shap
import matplotlib.pyplot as plt
import np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
        target_column="tothlos", 
        test_size=0.2, 
        random_state=42, 
        regression_models=None
    ):
        """
        Performs a GridSearchCV over multiple regression models defined in 'regression_models'.
        Returns the best model and its evaluation metrics.

        :param data: DataFrame containing features and target.
        :param target_column: The column to predict.
        :param test_size: Proportion of the data to use as the test set.
        :param random_state: Random seed for reproducibility.
        :param regression_models: Dictionary of regression models and their param grids.
        :return: (best_estimator, best_metrics, best_model_name)
        """
        if regression_models is None or not isinstance(regression_models, dict):
            raise ValueError("You must provide a valid dictionary of regression models and their param grids.")

        # 1. Adjust for binary targets
        data = self.adjust_for_binary_targets(data, target_column)

        # 2. Split data
        X_train, X_test, y_train, y_test = self.prepare_train_test(data, target_column, test_size, random_state)
        
        best_estimator = None
        best_score = -np.inf
        best_model_name = None
        best_metrics = {}

        # 3. Iterate over each model and run GridSearch
        for model_name, model_info in regression_models.items():
            print(f"\n--- Grid Search for {model_name} ---")
            base_regressor = model_info["regressor"]
            param_grid = model_info["param_grid"]

            # Create a pipeline for convenience (optional: add scaling if desired)
            pipeline = Pipeline([
                ("regressor", base_regressor)
            ])

            # Initialize GridSearchCV
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring="r2",       # Use R² for scoring (can also use "neg_mean_squared_error", etc.)
                cv=3,              # 3-fold cross-validation
                n_jobs=-1,         # Use all available cores
                verbose=1
            )

            # Fit GridSearch
            grid_search.fit(X_train, y_train)
            print(f"Best params for {model_name}: {grid_search.best_params_}")
            print(f"Best CV score for {model_name}: {grid_search.best_score_:.3f}")

            # Evaluate on test set
            best_model = grid_search.best_estimator_
            test_score = best_model.score(X_test, y_test)
            y_pred = best_model.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2_val = r2_score(y_test, y_pred)

            print(f"Test Score (R²) for {model_name}: {test_score:.3f}")
            print(f"MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2_val:.2f}")

            # Keep track of the best model overall (by test R²)
            if test_score > best_score:
                best_score = test_score
                best_estimator = best_model
                best_model_name = model_name
                best_metrics = {
                    "MAE": mae,
                    "MSE": mse,
                    "R2": r2_val,
                    "BestParams": grid_search.best_params_
                }

        print(f"\n=== Best Overall Model: {best_model_name} ===")
        print(f"R² on test set: {best_score:.3f}")
        print(f"MAE: {best_metrics['MAE']:.2f}, MSE: {best_metrics['MSE']:.2f}, R²: {best_metrics['R2']:.2f}")
        print(f"Best Params: {best_metrics['BestParams']}")

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
        import pdb;pdb.set_trace()
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)

        # Debug: Check SHAP values
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"Example SHAP values: {shap_values[:5]}")

        # Debug: Check feature importance
        feature_importances = rf.feature_importances_
        print("Top 10 Feature Importances:")
        for feature, importance in sorted(zip(X_test.columns, feature_importances), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feature}: {importance:.4f}")

        # Debug: Check for variability in SHAP values
        shap_variability = shap_values.var(axis=0)
        print(f"SHAP variability: {shap_variability}")

        # Display SHAP beeswarm plot
        print("Displaying SHAP beeswarm plot...")
        shap.summary_plot(shap_values, X_test, plot_type="dot")

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
    def generate_training_dataframe(self):
        """
        Generates a DataFrame for training with robust feature engineering.
        Handles binary, categorical, and numeric columns and ensures compatibility with ML models.
        """
        data = self.load_and_clean_data()
        if data.empty:
            print("No data available to process.")
            return pd.DataFrame()

        # Convert column names to lowercase for consistency
        data.columns = data.columns.str.lower()

        # Step 1: Binary columns
        binary_columns = [
            "sex", "ethnicity_hispanic", "transfus", "inout", "dnr", "prem_birth", 
            "ventilat", "asthma", "oxygen_sup", "tracheostomy", "stillinhosp", 
            "death30yn", "oxygen_at_discharge"
        ]
        data = ScoliosisFeatureEngineeringService.encode_binary(data, binary_columns)

        # Step 2: Categorical columns (apply one-hot encoding or grouping)
        categorical_columns = [
            "race", "surgspec", "delivery_mode", "wndclas", "asaclas", "anestech", "birth_location", "scolitype"
        ]
        data = ScoliosisFeatureEngineeringService.encode_categorical(data, categorical_columns, max_categories=5)

        # Step 3: Handle missing values
        data = ScoliosisFeatureEngineeringService.handle_missing(data)

        # Step 4: Add derived features
        data = ScoliosisFeatureEngineeringService.add_derived_features(data)

        # Step 5: Remove low-variance columns
        data = ScoliosisFeatureEngineeringService.drop_low_variance_columns(data)

        # Step 6: Remove perfectly correlated columns with the target
        data = ScoliosisFeatureEngineeringService.drop_perfectly_correlated_columns(data, target_column="tothlos")

        # Step 7: Ensure numeric values are within float32 range
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].clip(
            lower=np.finfo(np.float32).min,
            upper=np.finfo(np.float32).max
        )

        # Step 8: Ensure compatibility with RandomForest models
        data = ScoliosisFeatureEngineeringService.make_rf_compatible(data)

        # Validation and final check
        if data.isna().any().any():
            print("Warning: NaN values found in DataFrame after processing.")

        print(f"Training DataFrame generated with shape: {data.shape}")
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
    def encode_binary(data, binary_columns):
        """
        Encodes binary columns as 0/1, handling lowercase column names and string values.
        """
        for col in binary_columns:
            col_lower = col.lower()
            if col_lower in data.columns:
                data[col_lower] = data[col_lower].str.lower().map({
                    "yes": 1, "no": 0, "female": 0, "male": 1, "unknown/not reported": np.nan
                }).astype(float)
        return data

    @staticmethod
    def encode_categorical(data, categorical_columns, max_categories=10):
        """
        One-hot encodes categorical columns with limited unique values.
        Groups rare categories into "Other" if necessary.
        """
        for col in categorical_columns:
            col_lower = col.lower()
            if col_lower in data.columns:
                unique_values = data[col_lower].nunique()
                if unique_values > max_categories:
                    # Group rare categories into "Other"
                    top_categories = data[col_lower].value_counts().nlargest(max_categories).index
                    data[col_lower] = data[col_lower].apply(lambda x: x if x in top_categories else "Other")
                # Apply one-hot encoding
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
    def drop_low_variance_columns(data, threshold=0.01):
        """
        Drops columns with very low variance.

        :param data: Input DataFrame.
        :param threshold: Variance threshold below which columns are dropped.
        :return: DataFrame with low-variance columns removed.
        """
        print("Checking for low-variance columns...")
        low_variance_cols = [col for col in data.columns if data[col].nunique() == 1]
        print(f"Low-variance columns to drop: {low_variance_cols}")

        # Drop low-variance columns
        data = data.drop(columns=low_variance_cols, errors="ignore")
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
        correlated_columns = ['doptodis','tothlos','htooday']
        correlated_columns.remove(target_column)  # Exclude the target itself
        
        # TODO: Remove hospital_stay day stuff then get rid of this line.
        

        print(f"Columns perfectly correlated with {target_column}: {correlated_columns}")

        # Drop correlated columns from the original DataFrame
        data = data.drop(columns=correlated_columns, errors="ignore")
        return data