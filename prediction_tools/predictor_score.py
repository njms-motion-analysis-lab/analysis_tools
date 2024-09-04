import pickle
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
import sqlite3
import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from factor_analyzer import FactorAnalyzer
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from ts_fresh_params import get_params_for_column, beeswarm_name, categorize_feature, PARAMS

class PredictorScore(LegacyBaseModel):
    table_name = "predictor_score"

    def __init__(self, id=None, classifier_name=None, score_type=None, matrix=None, classifier_id=None, predictor_id=None, created_at=None, updated_at=None, multi_predictor_id=None):
        self.id = id
        self.classifier_name = classifier_name
        self.score_type = score_type
        self.matrix = matrix
        self.classifier_id = classifier_id
        self.predictor_id = predictor_id
        self.multi_predictor_id = multi_predictor_id


    def set_shap_matrix(self, aggregated_shap_values, combined_X):
        # Convert combined_X to numpy
        combined_X_np = combined_X.values

        # Serialize both matrices with the shape of aggregated_shap_values
        combined_matrix = (aggregated_shap_values, combined_X_np, combined_X.columns)
        self.matrix = pickle.dumps(combined_matrix)

        self.update(
            matrix = memoryview(self.matrix)
        )
    
    def view_dep_plot(self):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)  # We're using the global `combined_X` variable for columns

        # Call shap.summary_plot
        shap.dependence_plot('grad_data__kurtosis_x', aggregated_shap_values_retrieved, combined_X_df)

    def view_shap_plot(self, title=None, show_plot=False, abs_val=False, non_norm=True):
        # Fetch and process data
        aggregated_shap_values_retrieved, combined_X_df = self.fetch_and_prepare_data()

        # Ensure the shapes of values and data match
        reshaped_values = self.reshape_shap_values(aggregated_shap_values_retrieved, combined_X_df)

        # Create SHAP explanation object
        explanation = self.create_shap_explanation(reshaped_values, combined_X_df)

        # Plot the SHAP beeswarm plot
        self.plot_shap_beeswarm(explanation, show_plot, title, abs_val, non_norm)

    def fetch_and_prepare_data(self):
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")
        combo_beeswarm_names = [beeswarm_name(col_name) for col_name in combined_X_np_cols]
        combined_X_df = pd.DataFrame(combined_X_np, columns=combo_beeswarm_names)
        return aggregated_shap_values_retrieved, combined_X_df

    def reshape_shap_values(self, shap_values, data_df):
        # Attempt to reshape the SHAP values to match the row count of data_df
        reshaped_values = shap_values.reshape(data_df.shape[0], -1)

        # Check if the reshaped SHAP values match the expected number of columns
        if reshaped_values.shape[1] > data_df.shape[1]:
            # If there's an extra column, check if it seems like an alignment issue
            print("Warning: Reshaped SHAP values have extra columns.")
            reshaped_values = reshaped_values[:, :data_df.shape[1]]
        elif reshaped_values.shape[1] < data_df.shape[1]:
            # If columns are missing, raise an error to avoid silent data loss
            raise ValueError(
                f"Reshaped SHAP values have fewer columns ({reshaped_values.shape[1]}) than expected ({data_df.shape[1]})."
            )

        print(f"Shape of reshaped_values: {reshaped_values.shape}")
        print(f"Shape of combined_X_df: {data_df.shape}")

        return reshaped_values

    def create_shap_explanation(self, shap_values, data_df):
        try:
            return shap.Explanation(values=shap_values, data=data_df)
        except Exception as e:
            print(f"Error creating SHAP Explanation: {e}")
            print(f"reshaped_values shape: {shap_values.shape}")
            print(f"combined_X_df shape: {data_df.shape}")
            raise

    def plot_shap_beeswarm(self, explanation, show_plot, title, abs_val, non_norm):
        plt.figure()
        plt.title(self.get_info())
        try:
            shap.plots.beeswarm(explanation, show=False)
        except Exception as e:
            print(f"Error plotting beeswarm: {e}")
            print(f"Explanation values shape: {explanation.values.shape}")
            print(f"Explanation data shape: {explanation.data.shape}")
            return
        
        if show_plot:
            plt.show()
        else:
            self.save_plot(title, abs_val, non_norm)
            self.save_csv(explanation, title, abs_val, non_norm)
        plt.close()

    def save_plot(self, title, abs_val, non_norm):
        directory_path = self.build_directory_path(title, abs_val, non_norm)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        final_filename = self.get_info() + '.png'
        filepath = os.path.join(directory_path, final_filename)
        plt.savefig(filepath, bbox_inches='tight')

    def save_csv(self, explanation, title, abs_val, non_norm):
        directory_path = self.build_directory_path(title, abs_val, non_norm)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        
        # Create a DataFrame from the explanation data
        df = pd.DataFrame({
            'feature': explanation.feature_names,
            'shap_value': explanation.values.mean(0)
        })
        
        # Sort the DataFrame by absolute SHAP values
        df = df.reindex(df['shap_value'].abs().sort_values(ascending=False).index)
        
        # Save the DataFrame as a CSV file
        csv_filename = self.get_info() + '.csv'
        csv_filepath = os.path.join(directory_path, csv_filename)
        df.to_csv(csv_filepath, index=False)

    def build_directory_path(self, title, abs_val, non_norm):
        base_path = "generated_pngs/shap_beeswarm/"
        if title:
            return os.path.join(base_path, title)
        else:
            if abs_val:
                base_path = os.path.join(base_path, "abs")
            if not non_norm:
                base_path = os.path.join(base_path, "non_norm")
        return base_path


    def get_matrix(self, matrix_name):
        # Assuming this is a method to deserialize the 'matrix' attribute
        # This should return aggregated_shap_values, combined_X_np, combined_X_np_cols
        return pickle.loads(getattr(self, matrix_name))

    def perform_factor_analysis(self, n_factors=5, rotation='varimax'):
        # Fetch the serialized data
        _, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")
        
        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)
        
        # Initialize factor analysis object
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='principal')
        fa.fit(combined_X_df)
        
        # Get the loading matrix (factor loadings)
        loadings = fa.loadings_
        return combined_X_df.columns, loadings

    def get_eigenvalues(self):
        # Fetch the serialized data
        _, combined_X_np, _ = self.get_matrix("matrix")

        # Perform PCA to capture all components (equivalent to the number of features)
        pca = PCA(n_components=min(combined_X_np.shape))
        pca.fit(combined_X_np)

        # Eigenvalues can be retrieved from the explained variance attributes
        eigenvalues = pca.explained_variance_

        # Return eigenvalues
        return eigenvalues


    def plot_factor_loadings(self, n_factors=5, rotation='varimax'):
        columns, loadings = self.perform_factor_analysis(n_factors, rotation)
        
        plt.figure(figsize=(10, n_factors))
        for i in range(n_factors):
            plt.subplot(n_factors, 1, i+1)
            plt.barh(columns, loadings[:, i])
            plt.title(f'Factor {i+1}')
        # plt.tight_layout()
        plt.show()

    def get_info(self):
        from importlib import import_module
        Predictor = import_module("prediction_tools.legacy_predictor").Predictor
        pr = Predictor.get(self.predictor_id)
        task = Task.get(pr.task_id).description
        name = Sensor.get(pr.sensor_id).name
        try:
            acc = pr.get_classifier_accuracies()[self.classifier_name]
            acc_str = str(acc)  # Get the first two digits of the accuracy score
        except KeyError:
            acc_str = 'N/A'

        return task + '_' + name + '_' + self.classifier_name + " Accuracy: " + acc_str
    
    

    def get_standardized_shap_values(self):
        # Retrieve SHAP values and corresponding feature names
        aggregated_shap_values_retrieved, *_, feature_names = self.get_matrix("matrix")
        
        # Check if the retrieved SHAP values array is 3D
        if len(aggregated_shap_values_retrieved.shape) == 3:
            # If 3D, we'll take the first column of the last dimension
            reshaped_shap_values = aggregated_shap_values_retrieved[:, :, 0]
        else:
            # If already 2D, use as is
            reshaped_shap_values = aggregated_shap_values_retrieved
        
        if reshaped_shap_values.shape[0] == len(feature_names):
            if len(feature_names) != reshaped_shap_values.shape[0]:
                reshaped_shap_values = reshaped_shap_values.T
        
        # At this point, reshaped_shap_values should have shape (n_features, 50)
        
        return reshaped_shap_values, feature_names


    def find_optimal_clusters(self, standardized_shap_values, quick=True):
        best_score = -1
        optimal_clusters = 1

        num_samples = standardized_shap_values.shape[0]

        for i in range(2, min(5, num_samples+1)):  # Ensure the range of clusters does not exceed the number of samples
            if quick:
                c_alg = KMeans(n_clusters=i, init='k-means++', max_iter=2500, n_init=500, random_state=42)
            else:
                c_alg = GaussianMixture(n_components=i, covariance_type='full', max_iter=1500, n_init=100, random_state=0)

            labels = c_alg.fit_predict(standardized_shap_values)

            # Check if the number of unique labels is sufficient
            if len(set(labels)) > 1:
                try:
                    score = silhouette_score(standardized_shap_values, labels)
                    if score > best_score:
                        best_score = score
                        optimal_clusters = i
                except ValueError:
                    continue

        if best_score == -1:
            print("Warning: Could not find a valid clustering configuration with more than 1 unique label.")
            optimal_clusters = 1
            best_score = 0.0

        print("BEST SCORE", best_score)
        return optimal_clusters

    def cluster_features_shap(self):
        # Get standardized SHAP values
        standardized_shap_values, feature_names = self.get_standardized_shap_values()
        
        # Apply scaling and PCA to the data
        scaler = StandardScaler()
        scaled_shap_values = scaler.fit_transform(standardized_shap_values)
        pca = PCA()
        pca.fit(scaled_shap_values)

        # Calculate the cumulative explained variance ratio
        cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

        # Find the number of components that capture the desired threshold of variance
        threshold = 0.95  # Adjust this value based on your desired threshold
        n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1

        # Print the number of components and the cumulative explained variance ratio
        print(f"Number of components: {n_components}")
        print(f"Cumulative explained variance ratio: {cumulative_variance_ratio[n_components - 1]:.2f}")

        # Apply PCA with the determined number of components
        pca = PCA(n_components=n_components)
        transformed_shap_values = pca.fit_transform(scaled_shap_values)
        
        # Find the optimal number of clusters based on transformed SHAP values
        c = self.find_optimal_clusters(transformed_shap_values, quick=True)
        
        # Define clustering algorithm (KMeans)
        kmeans = KMeans(n_clusters=c, init='k-means++', max_iter=5000, n_init=500, random_state=42)
        # temp try gaussian mixture
        # kmeans = GaussianMixture(n_components=optimal_clusters, covariance_type='full', max_iter=1500, n_init=100, random_state=0)
        clusters = kmeans.fit_predict(transformed_shap_values)
        # Pair each feature name with its cluster label
        feature_cluster_pairs = list(zip(feature_names, clusters))
        
        return feature_cluster_pairs
    
    def plot_3d_pca(self):
        # Fetch the serialized data
        _, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Perform PCA
        pca = PCA(n_components=3)  # We want to project down to 3 dimensions
        components = pca.fit_transform(combined_X_df)

        # Get clustering results
        feature_cluster_pairs = self.cluster_features_shap()
        feature_to_cluster = {feature: cluster for feature, cluster in feature_cluster_pairs}

        # Create a list of cluster labels aligned with the order of columns in PCA
        cluster_labels = [feature_to_cluster.get(feature, -1) for feature in combined_X_df.columns]

        # Plotting
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=cluster_labels[:len(components)], cmap='viridis', marker='o')
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        # Create a color bar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster Label')

        plt.title('3D PCA of Features')
        plt.show()

    def view_shap_heatmap(self, title=None, abs_val=False):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Convert the retrieved SHAP values and feature values into an Explanation object
        explanation = shap.Explanation(values=aggregated_shap_values_retrieved, data=combined_X_df, feature_names=combined_X_np_cols)
        # Make sure directory exists or create it
        if not abs_val:
            directory_path = "generated_pngs/shap_heatmap/"
        else:
            directory_path = "generated_pngs/shap_heatmap/abs"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Construct the filename from the provided title
        filename = os.path.join(directory_path, title.replace(" ", "_") + ".png")

        # Create a new figure explicitly
        plt.figure()
        # Debug: print the shape and properties of the Explanation object
        print("Explanation values shape:", explanation.values.shape)
        print("Explanation feature names:", explanation.feature_names)

        # Call shap.summary_plot with the Explanation object
        shap.plots.heatmap(explanation, show=False)
        
        # Save the current active figure as PNG
        plt.savefig(filename, bbox_inches='tight')
        
        # Optionally, close the plot to free up resources
        plt.close()
            
    def get_top_n_features(self, n):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        print("AGGREGATED_SHAP_VALUES_RETRIEVED", aggregated_shap_values_retrieved)
        print("COMBINED_X_NP", combined_X_np)
        print("COMBINED_X_NP_COLS", combined_X_np_cols)

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Check if the retrieved SHAP values array is 3D
        if len(aggregated_shap_values_retrieved.shape) == 3:
            # Sum the absolute values across the third dimension
            reshaped_shap_values = np.sum(np.abs(aggregated_shap_values_retrieved), axis=2)

            # Ensure that reshaping matches the feature space
            if reshaped_shap_values.shape[1] != len(combined_X_np_cols):
                raise ValueError("The reshaped SHAP values do not match the number of columns in the combined data.")
        else:
            reshaped_shap_values = np.abs(aggregated_shap_values_retrieved)

        # Convert the reshaped SHAP values into a DataFrame
        shap_values_df = pd.DataFrame(reshaped_shap_values, columns=combined_X_np_cols)

        # Calculate the mean absolute SHAP value for each feature
        mean_abs_shap = shap_values_df.mean().sort_values(ascending=False)

        # Get the top N features
        top_features = mean_abs_shap.head(n).index.tolist()
        
        return top_features

    # Define the methods to get top categories, aggregate SHAP values by category, and normalize those values.
    def get_top_feature_categories(self, feature_names, axis=False, n=100):
        categorized_features = [(feature, categorize_feature(feature, axis)) for feature in feature_names[:n]]
        category_counts = {}
        for _, category in categorized_features:
            category_counts[category] = category_counts.get(category, 0) + 1
        top_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:n]
        return top_categories

    def aggregate_shap_values_by_category(self, shap_values, axis=False):
        category_shap_values = {}
        
        for feature, shap_value in shap_values.items():
            category = categorize_feature(feature, axis)
            category_shap_values[category] = category_shap_values.get(category, 0) + abs(shap_value)
        return category_shap_values

    def normalize_shap_values(self, category_shap_values):
        total_shap_value = sum(category_shap_values.values())
        normalized_shap_values = {category: value / total_shap_value for category, value in category_shap_values.items()}
        return normalized_shap_values



    def get_shap_values_for_features(self, features_list):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Check if the retrieved SHAP values array is 3D
        if len(aggregated_shap_values_retrieved.shape) == 3:
            # Sum the absolute values across the third dimension to maintain correct shape
            reshaped_shap_values = np.sum(np.abs(aggregated_shap_values_retrieved), axis=2)

            # Ensure that reshaped SHAP values match the number of features
            if reshaped_shap_values.shape[1] != len(combined_X_np_cols):
                raise ValueError("Mismatch between SHAP values and feature columns.")
        else:
            reshaped_shap_values = np.abs(aggregated_shap_values_retrieved)

        # Convert the reshaped SHAP values into a DataFrame with correct columns
        shap_values_df = pd.DataFrame(reshaped_shap_values, columns=combined_X_np_cols)

        # Initialize a dictionary to store mean absolute SHAP values for specified features
        mean_abs_shap_values = {}

        # Iterate over the features list to fetch mean absolute SHAP values
        for feature in features_list:
            if feature in shap_values_df.columns:
                # Calculate the mean absolute SHAP value for the feature
                mean_abs_shap_values[feature] = np.abs(shap_values_df[feature]).mean()
            else:
                # Assign 0.000 if the feature is not in the DataFrame
                mean_abs_shap_values[feature] = 0.000

        return mean_abs_shap_values
