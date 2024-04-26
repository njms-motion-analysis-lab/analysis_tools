import pickle
from sklearn.metrics import silhouette_score
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
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        # combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)
        combo_beeswarm_names = []
        for col_name in combined_X_np_cols:
            combo_beeswarm_names.append(beeswarm_name(col_name))

        combined_X_df = pd.DataFrame(combined_X_np, columns=combo_beeswarm_names)
        


        # Convert the retrieved SHAP values and feature values into an Explanation object
        explanation = shap.Explanation(values=aggregated_shap_values_retrieved, data=combined_X_df)

        # Create a new figure explicitly
        plt.figure()

        # Call shap.summary_plot with the Explanation object
        shap.plots.beeswarm(explanation, show=False)

        if show_plot:
            # Display the plot
            plt.show()
        else:
            directory_path = "generated_pngs/shap_beeswarm/"
            # Make sure directory exists or create it
                    # Make sure directory exists or create it
            if title is not None:
                directory_path = (directory_path + title + "/")
            else:

                if not abs_val:
                    directory_path = "generated_pngs/shap_beeswarm/"
                else:
                    directory_path = "generated_pngs/shap_beeswarm/abs/"
                
                if non_norm == False:
                    directory_path = "generated_pngs/shap_beeswarm/non_norm/"
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            final = self.get_info() + '.png'
            filename = os.path.join(directory_path, final)

            # Save the current active figure as PNG
            print("HIIIIII")
            print(filename)
            plt.savefig(filename, bbox_inches='tight')
        
        # Optionally, close the plot to free up resources
        plt.close()

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
        return task + '_' + name + '_' + self.classifier_name
    
    def get_standardized_shap_values(self):
        # Retrieve SHAP values and corresponding feature names
        aggregated_shap_values_retrieved, _, feature_names = self.get_matrix("matrix")
        
        # Standardize the SHAP values; remember to transpose for standardization if necessary
        scaler = StandardScaler()
        standardized_shap_values = scaler.fit_transform(aggregated_shap_values_retrieved.T)  # Assuming features are columns
        
        return standardized_shap_values, feature_names
    
    def find_optimal_clusters(self, standardized_shap_values):
        # Use silhouette score to find the optimal number of clusters
        best_score = -1
        optimal_clusters = 2  # Default in case the loop doesn't find a better number

        for i in range(2, 11):  # Starting from 2 because silhouette score cannot be calculated with a single cluster
            kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            labels = kmeans.fit_predict(standardized_shap_values)
            score = silhouette_score(standardized_shap_values, labels)
            
            if score > best_score:
                best_score = score
                optimal_clusters = i

        return optimal_clusters

    def cluster_features_shap(self):
        # Get standardized SHAP values
        standardized_shap_values, feature_names = self.get_standardized_shap_values()

        # Apply scaling and PCA to the data
        scaler = StandardScaler()
        scaled_shap_values = scaler.fit_transform(standardized_shap_values)
        pca = PCA(n_components=0.95)  # Adjust the number of components as needed
        transformed_shap_values = pca.fit_transform(scaled_shap_values)

        # Find the optimal number of clusters based on transformed SHAP values
        optimal_clusters = self.find_optimal_clusters(transformed_shap_values)

        # Define clustering algorithms to try
        clustering_algorithms = [
            KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=1000, n_init=50, random_state=0),
            DBSCAN(eps=0.5, min_samples=5),
            AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward')
        ]

        best_score = -1
        best_clusters = None
        best_algorithm = None

        # Try each clustering algorithm and select the one with the highest silhouette score
        for algorithm in clustering_algorithms:
            clusters = algorithm.fit_predict(transformed_shap_values)
            
            # Check if only one cluster is found
            if len(np.unique(clusters)) == 1:
                score = -1
            else:
                score = silhouette_score(transformed_shap_values, clusters)

            if score > best_score:
                best_score = score
                best_clusters = clusters
                best_algorithm = algorithm

        # Pair each feature name with its cluster label
        feature_cluster_pairs = list(zip(feature_names, best_clusters))

        print(f"Best clustering algorithm: {type(best_algorithm).__name__}")

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
        explanation = shap.Explanation(values=aggregated_shap_values_retrieved, data=combined_X_df, 
                               feature_names=combined_X_np_cols)
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

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Convert the retrieved SHAP values into a DataFrame
        shap_values_df = pd.DataFrame(aggregated_shap_values_retrieved, columns=combined_X_np_cols)

        # Calculate the mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values_df).mean().sort_values(ascending=False)

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

        # Convert the retrieved SHAP values into a DataFrame
        shap_values_df = pd.DataFrame(aggregated_shap_values_retrieved, columns=combined_X_np_cols)

        # Initialize a dictionary to store mean absolute SHAP values
        mean_abs_shap_values = {}

        # Iterate over the features list
        for feature in features_list:
            if feature in shap_values_df.columns:
                # Calculate the mean absolute SHAP value for the feature
                mean_abs_shap_values[feature] = np.abs(shap_values_df[feature]).mean()
            else:
                # Assign 0.000 if the feature is not in the DataFrame
                mean_abs_shap_values[feature] = 0.000

        return mean_abs_shap_values