# import sqlite3

from collections import defaultdict
import collections
import csv
import os
import re
from sklearn.impute import SimpleImputer
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
import json
import pickle
import pandas as pd
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from models.legacy_cohort import Cohort
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from prediction_tools.legacy_predictor import Predictor

import matplotlib.colors as mcolors
from matplotlib import colors as mcolors 
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from datetime import datetime, timedelta
from prediction_tools.predictor_score import PredictorScore
from ts_fresh_params import fn_get_params_for_column
from viewers.matrix_plotter import MatrixPlotter
from viewers.multi_plotter import MultiPlotter
import matplotlib.colors as mcolors



SENSOR_CODES = [
    'rfin_x',
    'rwrb_x',
    'rwra_x',
    'rfrm_x',
    'relb_x',
    # 'relbm_x',
    'rupa_x',
    'rsho_x',
    'rbhd_x',
    'rfhd_x',
]

TOP_TREE_MODELS = ['XGBoost']


# TOP_TREE_MODELS = ['XGBoost']

NUM_TOP = 1000
class MultiPredictor(LegacyBaseModel):
    table_name = "multi_predictor"

    def __init__(self, id=None, task_id=None, sensors=Sensor.where(name=SENSOR_CODES), model="default", items = None, created_at=None, updated_at=None, cohort_id=None):
        super().__init__()
        self.id = id
        self.task_id = task_id
        self.sensors = sensors
        self.model = model
        self.items = {}
        self.created_at = created_at
        self.updated_at = updated_at
        self.cohort_id = cohort_id
        self.non_norm = True,
        self.abs_val = False,
        self.curr_axis = None,
        self.elbow = None,
        

    def gen_items_for_sensors(self, sensors=None, ntaf=False):
        if sensors is None:
            sensors = self.get_sensors()
        # for sen in sensors:
        #     self.gen_items_for_sensor(sen, ntaf)

    def task(self):
        return Task.get(self.task_id)

    def gen_items_for_sensor(self, snr=None, ntaf=True):
        if not ntaf:
            print(f"Generating normed, abs score, for task: {self.task().description}")
            nfat = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=False, cohort_id=self.cohort_id, multi_predictor_id=self.id)
            nfat.train_from(use_shap=True)
            print(f"Generating normed, non abs score, for task: {self.task().description}")
            nfaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=False, cohort_id=self.cohort_id, multi_predictor_id=self.id)
            nfaf.train_from(use_shap=True)
            print(f"Generating non normed, abs score, for task: {self.task().description}")
        
        ntaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=True, abs_val=True, cohort_id=self.cohort_id, multi_predictor_id=self.id)
        ntaf.train_from(use_shap=True)
        return ntaf
    
    def gen_predictors_for_sensor(self, snr):
        Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=False, cohort_id=self.cohort_id, multi_predictor_id=self.id)
    
    def get_sensors(self):
        return Sensor.where(name=SENSOR_CODES)

    def get_predictors(self, abs_val=False, non_norm=True):
        return Predictor.where(multi_predictor_id=self.id, abs_val=abs_val, non_norm=non_norm)

    get_preds = get_predictors

    def get_norm_predictors(self):
        return Predictor.where(multi_predictor_id=self.id, non_norm=0)

    get_norm_preds = get_norm_predictors

    def get_abs_predictors(self):
        return Predictor.where(multi_predictor_id=self.id, abs_val=True, non_norm=1)

    get_abs_preds = get_abs_predictors

    def get_predictor_model_accuracies(self, model_name, abs_val=False, non_norm=True):
        preds = self.get_predictors(abs_val=abs_val, non_norm=non_norm)
        combos = []
        for pr in preds:
            print("NUM PREDS for", model_name, Sensor.get(pr.sensor_id).name, len(pr.get_predictor_scores(model_name=model_name)))
            if pr.get_accuracies() != {}:
                combos.append(
                    [
                        pr.get_predictor_scores(model_name=model_name), 
                        pr.get_accuracies()['classifier_accuracies'][model_name],
                        Sensor.get(pr.sensor_id)
                    ]
                )
            else:
                print("Empty Predictor Accuracies:", pr.attrs())
        print("combos:", combos)
        return combos


    def get_predictor_scores_for_model(self, model_name, abs_val=False, non_norm=True, sort_by_sensor=False, reverse_sensor_order=False):
        preds = self.get_predictor_model_accuracies(model_name=model_name, abs_val=abs_val, non_norm=non_norm)

        print("PRED LEN", len(preds))
        preds = [pred for pred in preds if pred[2].name in SENSOR_CODES]
        if sort_by_sensor:
            # Create a mapping of sensor names to their order in SENSOR_CODES
            sensor_order = {sensor_name: i for i, sensor_name in enumerate(SENSOR_CODES)}
            print(sensor_order)
            if reverse_sensor_order:
                # Reverse the sensor order
                max_index = len(SENSOR_CODES) - 1
                sensor_order = {sensor_name: max_index - i for i, sensor_name in enumerate(SENSOR_CODES)}

            # Sort by the order of sensors as per SENSOR_CODES, then by accuracy score
            preds.sort(key=lambda x: (sensor_order.get(x[2].name, float('inf')), -x[1]))
        else:
            # Sort by accuracy score in descending order
            preds.sort(key=lambda x: x[1], reverse=True)

        print("done first")
        return preds

    def show_abs_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
        return self.show_predictor_scores(
            models=models,
            abs_val=True, 
            non_norm=True,
            reverse_order=reverse_order, 
            first_model_features=first_model_features, 
            num_top=num_top, 
            use_cat=use_cat, 
            axis=axis,
            include_accuracy=include_accuracy,
        )

    def show_norm_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=True):
        return self.show_predictor_scores(
            models,
            abs_val=False, 
            non_norm=False,
            reverse_order=reverse_order, 
            first_model_features=first_model_features, 
            num_top=num_top, 
            use_cat=use_cat, 
            axis=axis,
            include_accuracy=include_accuracy,
        )

    def show_combo_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=True):
        return self.show_predictor_scores(
            models,
            abs_val=False,
            non_norm=True,
            reverse_order=reverse_order, 
            first_model_features=first_model_features, 
            num_top=num_top, 
            use_cat=use_cat, 
            axis=axis,
            include_accuracy=include_accuracy,
        )

    def show_all_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=True):
        return self.show_predictor_scores(
            models,
            abs_val=False, 
            non_norm=False,
            reverse_order=reverse_order, 
            first_model_features=first_model_features, 
            num_top=num_top, 
            use_cat=use_cat, 
            axis=axis,
            include_accuracy=include_accuracy,
        )
    
    def show_default_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=True):
        return self.show_predictor_scores(
            models,
            abs_val=False, 
            non_norm=True, 
            reverse_order=reverse_order,
            first_model_features=first_model_features, 
            num_top=num_top, 
            use_cat=use_cat, 
            axis=axis,
            include_accuracy=include_accuracy,
        )
    

    
    def linkage_mat(self, sim_matrix=None, view=True, abs_val=False, non_norm=True, model=None, save=False, file_name="block_dendrogram_clusters.csv"):
        if sim_matrix is None:
            sim_matrix, unique_features = self.sim_matrix(abs_val=abs_val, non_norm=non_norm, model=model)
        
        # Generate the linkage matrix
        Z = linkage(1 - sim_matrix, 'ward')  # Use 1-similarity as distance
        
        # Find the optimal color threshold based on the largest difference in heights
        heights = Z[:, 2]
        diffs = np.diff(heights)
        color_threshold = heights[np.argmax(diffs)]
        
        # Create a bigger figure
        plt.figure(figsize=(24, 12))  # Adjust width and height as needed
        
        # Create the dendrogram
        dendro = dendrogram(Z, labels=unique_features, color_threshold=color_threshold)
        
        # Adjust space below the dendrogram
        plt.subplots_adjust(bottom=0.2)  # Adjust the bottom margin as needed (0.2 adds more space)
        
        # Create a legend for the colors
        legend_handles = []
        color_map = {}
        for color in sorted(set(dendro['color_list'])):
            # if color != 'C0':  # Exclude the color if it's the default blue used for the last p merges
            legend_handles.append(plt.Line2D([0], [0], color=color, lw=4))
            color_map[color] = mcolors.to_hex(color)
        
        plt.legend(legend_handles, [f"Cluster {i + 1}" for i in range(len(legend_handles))], loc='upper right')
        
        plt.title("Dendrogram of Clustered Features")
        plt.xlabel("Features")
        plt.ylabel("Distance")
        
        if view:
            plt.show()
        else:
            if save:
                # Save the figure with specified filename, DPI, and layout
                curr_name = file_name.replace('.csv', '.png')
                cohort_name = self.cohort_name()
                filename = cohort_name + "_" + curr_name
                if self.model is not None:
                    filename = self.model + "_" + filename
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Saved dendrogram figure as {file_name.replace('.csv', '.png')}")
        
        # Extract the color information
        leaf_colors = dendro['color_list']
        leaf_labels = dendro['ivl']
        
        # Map from labels to colors
        label_color_map = {label: color for label, color in zip(leaf_labels, leaf_colors)}
        
        if save:
            # Prepare the data for saving
            df = pd.DataFrame(list(label_color_map.items()), columns=['Feature', 'Cluster Color'])
            df.to_csv(file_name, index=False)
            print(f"Saved cluster data to {file_name}")
        
        return label_color_map, color_map


    def sim_matrix(self, feature_cluster_map=None, model=None, non_norm=False, abs_val=False):
        print("ABS VAL", abs_val, "non_norm", non_norm)
        if feature_cluster_map is None:
            feature_cluster_map = self.feature_cluster_map(non_norm=non_norm, abs_val=abs_val, model=model)

        unique_features = list(feature_cluster_map.keys())
        num_features = len(unique_features)

        # Initialize the similarity matrix with zeros
        similarity_matrix = np.zeros((num_features, num_features))
        # Fill the similarity matrix
        for i, feature_i in enumerate(unique_features):
            for j, feature_j in enumerate(unique_features):
                if i >= j:  # Avoid repeating calculations and ensure diagonal is 0
                    continue
                # Flatten the list of clusters for both features and then calculate intersection
                clusters_i = [cluster for clusters in feature_cluster_map[feature_i].values() for cluster in clusters]
                clusters_j = [cluster for clusters in feature_cluster_map[feature_j].values() for cluster in clusters]
                
                # Calculate similarity as the number of times feature_i and feature_j are in the same cluster
                similarity = sum(1 for x in clusters_i if x in clusters_j)
                # Normalize by the number of models/sensors they were both part of
                total_i = len(clusters_i)
                total_j = len(clusters_j)
                normalized_similarity = similarity / max(total_i, total_j) if max(total_i, total_j) > 0 else 0
                
                similarity_matrix[i, j] = normalized_similarity
                similarity_matrix[j, i] = normalized_similarity  # Symmetric matrix

        return similarity_matrix, unique_features

    def feature_cluster_map(self, non_norm=True, abs_val=False, model=None, new=True):
        if new:
            sensor_output = self.aggregate_shap_values_2(non_norm=non_norm, abs_val=abs_val, models=[model])
        else:
            sensor_output = self.aggregate_shap_values(non_norm=non_norm, abs_val=abs_val, models=[model])


        feature_cluster_map = {}

        for model, sensors in sensor_output.items():
            for feature, sensor_clusters in sensors.items():
                if feature not in feature_cluster_map:
                    feature_cluster_map[feature] = {}
                for sensor_id, cluster in sensor_clusters:
                    if sensor_id not in feature_cluster_map[feature]:
                        feature_cluster_map[feature][sensor_id] = []
                    feature_cluster_map[feature][sensor_id].append(cluster)

        return feature_cluster_map

    def aggregate_shap_values(self, models=TOP_TREE_MODELS, abs_val=False, non_norm=True, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
        pred_kit = {}
        pred_sensor_kit = {}
        for m in models:
            pred_kit[m] = self.get_predictor_scores_for_model(m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm)
            # Ensure pred_sensor_kit[m] is initialized as a dictionary for the model
            pred_sensor_kit[m] = {}
            for predictor_score in pred_kit[m]:
                # Perform clustering for each sensor and model
                try:
                    cluster_assignments = predictor_score[0][0].cluster_features_shap()
                except IndexError:
                    import pdb;pdb.set_trace()
                # Assuming sensor_object has a unique identifier, such as an ID or a name
                sensor_id = predictor_score[-1].id
                # Adjust the structure of pred_sensor_kit to include sensor information
                try:
                    for feature, cluster in cluster_assignments:
                        if feature not in pred_sensor_kit[m]:
                            pred_sensor_kit[m][feature] = []
                        pred_sensor_kit[m][feature].append((sensor_id, cluster))
                except UnboundLocalError:
                    import pdb;pdb.set_trace()
        return pred_sensor_kit
    
    def prepare_sensor_data(self, models=["RandomForest"], abs_val=False, non_norm=False, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False, metric=None):
        pred_kit = {}
        pred_sensor_kit = {}
        for m in models:
            pred_kit[m] = self.get_predictor_scores_for_model(m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm)
            pred_sensor_kit[m] = {}

            # Collect SHAP values and feature names from all sensors for this model
            all_shap_values = []
            all_feature_names = []
            all_sensor_ids = []

            for predictor_score in pred_kit[m]:
                shap_values, feature_names = predictor_score[0][0].get_standardized_shap_values()
                sensor = predictor_score[-1]
                # Extend feature names list
                all_feature_names.extend([f"{sensor.id}_{name}" for name in feature_names])
                all_shap_values.append(shap_values)
                all_sensor_ids.extend([sensor.id] * len(feature_names))

            # Create a DataFrame with all SHAP values
            combined_data = pd.DataFrame(np.concatenate(all_shap_values, axis=1), 
                                        columns=all_feature_names)
            # Sort by sensor ID if reverse_order is specified
            if reverse_order:
                combined_data = combined_data.sort_index(axis=1, key=lambda x: [-int(col.split('_')[0]) for col in x])
            
            # Calculate distance matrix using Euclidean distance
            if metric is None:
                metric = 'euclidean'
            distance_matrix = pdist(combined_data.T, metric=metric)
            distance_matrix = squareform(distance_matrix)
            
        return combined_data, distance_matrix, all_sensor_ids

    def save_shap_data_to_csv(self, combined_data, filename='shap_data_with_info.csv'):
        """
        Save the SHAP data to a CSV file with additional feature information.
        
        :param combined_data: pandas DataFrame containing the SHAP values
        :param filename: str, name of the output CSV file
        """
        # Create a list to store feature information
        feature_info = []

        # Iterate through each feature (column) in the DataFrame
        for feature in combined_data.columns:
            params = fn_get_params_for_column(feature)
            if params:
                feature_info.append({
                    'Feature': feature,
                    'Sensor ID': params['sensor_id'],
                    'Sensor Code': params['sensor_code'],
                    'Feature Name': params['feature_name'],
                    'Feature Index': params['feature_index'],
                    'Axis': params['axis'],
                    'Parameters': str(params['params'])  # Convert to string for CSV compatibility
                })
            else:
                feature_info.append({
                    'Feature': feature,
                    'Sensor ID': 'N/A',
                    'Sensor Code': 'N/A',
                    'Feature Name': 'N/A',
                    'Feature Index': 'N/A',
                    'Axis': 'N/A',
                    'Parameters': 'N/A'
                })

        # Create a DataFrame with feature information
        feature_info_df = pd.DataFrame(feature_info)

        # Transpose the original data for better CSV format
        transposed_data = combined_data.T

        # Merge feature information with transposed data
        result = pd.merge(feature_info_df, transposed_data, left_on='Feature', right_index=True)

        # Save to CSV
        filename = self.cohort_name() + filename
        if self.model is not None:
            filename = self.model + '_' + filename
        result.to_csv(filename, index=False)
        print(f"CSV file '{filename}' has been created successfully.")

    
    

    def check_time_predictors(self):
        predictors = self.get_preds()

    def plot_shapley_over_time_updated(self, averaged_shap_scores, feature_cluster_map, color_mapping, averaged_sensors):
        
        
        plt.figure(figsize=(15, 10))
        
        sensor_names = averaged_sensors
        clusters = set(feature_cluster_map.values())
        
        cluster_shap_scores = {cluster: {sensor: 0 for sensor in sensor_names} for cluster in clusters}
        
        # Create a mapping between shortened feature names and full feature names
        feature_name_mapping = {}
        for full_feature_name in feature_cluster_map.keys():
            short_name = re.sub(r'^\d+_grad_data__', '', full_feature_name)
            feature_name_mapping[short_name] = full_feature_name
        
        print("Debug: Processing Shapley scores")
        for sensor_idx, sensor_score_set in enumerate(averaged_shap_scores):
            sensor_name = sensor_names[sensor_idx]
            print(f"  Sensor: {sensor_name}")
            for feature, shap_value in sensor_score_set.items():
                if feature != 'ave_accuracy' and shap_value != 0:
                    if feature in feature_name_mapping:
                        full_feature_name = feature_name_mapping[feature]
                        cluster = feature_cluster_map[full_feature_name]
                        cluster_shap_scores[cluster][sensor_name] += abs(shap_value)
                        print(f"    Feature: {full_feature_name}, Cluster: {cluster}, Shapley value: {shap_value}")
                    else:
                        print(f"    Warning: No cluster found for feature: {feature}")
        
        print("\nDebug: Cluster Shapley Scores")
        for cluster, scores in cluster_shap_scores.items():
            print(f"Cluster {cluster}:")
            for sensor, score in scores.items():
                print(f"  {sensor}: {score}")
        
        all_values = [score for scores in cluster_shap_scores.values() for score in scores.values()]
        min_non_zero = min([v for v in all_values if v > 0]) if any(v > 0 for v in all_values) else 1e-20
        print(f"\nMin non-zero value: {min_non_zero}")
        
        for cluster, scores in cluster_shap_scores.items():
            sensors_list = list(scores.keys())
            values_list = list(scores.values())
            
            # Use a small constant to avoid log(0), but keep it smaller than the smallest non-zero value
            values_list = [max(v, min_non_zero / 10) for v in values_list]
            
            plt.semilogy(sensors_list, values_list, color=color_mapping[cluster], label=f'Cluster {cluster}', marker='o')
            print(f"\nPlotting Cluster {cluster}:")
            print(f"  Sensors: {sensors_list}")
            print(f"  Values: {values_list}")
        
        plt.title('Shapley Values Over Time by Cluster')
        plt.xlabel('Sensors')
        plt.ylabel('Cumulative Absolute Shapley Value (log scale)')
        plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Set y-axis limits to focus on the range of the data
        plt.ylim(bottom=min_non_zero / 100, top=max(all_values) * 10)
        
        plt.tight_layout()
        plt.show()


    def create_dendrogram_with_clusters(self, combined_data, distance_matrix, method='ward', alt_metric=None):
        linked = linkage(distance_matrix, method=method)

        
        # Determine number of clusters
        # If self.elbow is not available, use the distance threshold method
        max_d = 0.7 * max(linked[:, 2])  # 70% of the maximum distance as a default threshold
        cluster_assignments = fcluster(linked, max_d, criterion='distance')
        n_clusters = len(np.unique(cluster_assignments))
        
        # Create a color mapping for clusters
        unique_clusters = np.unique(cluster_assignments)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        color_mapping = {cluster: mcolors.to_hex(color) for cluster, color in zip(unique_clusters, colors)}

        
        # Function to get the color for a given link
        def link_color_func(link):
            if link > len(cluster_assignments) - 2:
                return '#808080'  # Gray color for links above the cut
            left = int(linked[link, 0])
            right = int(linked[link, 1])
            if left < len(cluster_assignments):
                cluster_left = cluster_assignments[left]
            else:
                cluster_left = cluster_assignments[left - len(cluster_assignments)]
            if right < len(cluster_assignments):
                cluster_right = cluster_assignments[right]
            else:
                cluster_right = cluster_assignments[right - len(cluster_assignments)]
            if cluster_left == cluster_right:
                return color_mapping[cluster_left]
            else:
                return '#808080'  # Gray color for links between different clusters
        
        # Create dendrogram
        plt.figure(figsize=(20, 10))  # Increased figure size for better readability
        ddata = dendrogram(
            linked,
            orientation='top',
            labels=combined_data.columns,
            distance_sort='descending',
            show_leaf_counts=True,
            link_color_func=link_color_func,
            leaf_font_size=8,
            color_threshold=None  # This allows our custom coloring
        )
        
        # Color the leaf labels based on their cluster assignment
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        # for lbl in xlbls:
        #     lbl.set_color(color_mapping[cluster_assignments[combined_data.columns.get_loc(lbl.get_text())]])
        title = f'Hierarchical Clustering Dendrogram ({n_clusters} clusters)'
        if self.model == "grad_set":
            title = title + "--Full Motion"
        title = self.cohort_name() + " " + title
        if alt_metric is not None:
            title = title + alt_metric
        plt.title(title)
        plt.xlabel('Feature')
        plt.ylabel('Distance')
        plt.xticks(rotation=90)
        
        # Add a horizontal line at the cut threshold if using distance criterion
        if self.elbow is None:
            plt.axhline(y=max_d, c='k', lw=1, linestyle='dashed')
        
        plt.tight_layout()
        
        # Add a legend for cluster colors
        legend_elements = [plt.Line2D([0], [0], color=color, lw=4, label=f'Cluster {cluster}')
                        for cluster, color in color_mapping.items()]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # plt.show()
        
        # Create a dictionary mapping features to their cluster assignments
        feature_cluster_map = dict(zip(combined_data.columns, cluster_assignments))
        color_mapping = self.set_color_mapping(feature_cluster_map)
        for lbl in xlbls:
            lbl.set_color(color_mapping[cluster_assignments[combined_data.columns.get_loc(lbl.get_text())]])
        # Save the figure
        save_dir = 'dendrograms'
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{title.replace(' ', '_')}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free up memory

        # Create a dictionary mapping features to their cluster assignments
        feature_cluster_map = dict(zip(combined_data.columns, cluster_assignments))
        return feature_cluster_map, color_mapping
    
    def aggregate_shap_values_2(self, models=TOP_TREE_MODELS, abs_val=False, non_norm=True, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
        pred_kit = {}
        pred_sensor_kit = {}
        
        for m in models:
            pred_kit[m] = self.get_predictor_scores_for_model(m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm)
            pred_sensor_kit[m] = {}
            
            # Collect SHAP values and feature names from all sensors for this model
            all_shap_values = []
            all_feature_names = []
            all_sensor_ids = []
            
            
            for predictor_score in pred_kit[m]:
                shap_values, feature_names = predictor_score[0][0].get_standardized_shap_values()
                sensor = predictor_score[-1]
                all_shap_values.append(shap_values)
                all_feature_names.extend([f"{sensor.id}_{name}" for name in feature_names])
                all_sensor_ids.extend([sensor.id] * len(feature_names))
            
            # Create a matrix of SHAP values, with NaN for missing features
            max_features = max(shap.shape[1] for shap in all_shap_values)
            combined_shap_values = np.full((sum(shap.shape[0] for shap in all_shap_values), max_features), np.nan)
            row_idx = 0
            for shap in all_shap_values:
                combined_shap_values[row_idx:row_idx+shap.shape[0], :shap.shape[1]] = shap
                row_idx += shap.shape[0]
            
            # Impute missing values
            imputer = SimpleImputer(strategy='mean')
            imputed_shap_values = imputer.fit_transform(combined_shap_values)
            
            # Apply scaling and PCA
            scaler = StandardScaler()
            scaled_shap_values = scaler.fit_transform(imputed_shap_values)
            pca = PCA()
            pca.fit(scaled_shap_values)
            
            # Calculate the cumulative explained variance ratio
            cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
            threshold = 0.95
            n_components = np.argmax(cumulative_variance_ratio >= threshold) + 1
            print(f"Number of components for model {m}: {n_components}")
            print(f"Cumulative explained variance ratio: {cumulative_variance_ratio[n_components - 1]:.2f}")
            
            # Apply PCA with the determined number of components
            pca = PCA(n_components=n_components)
            transformed_shap_values = pca.fit_transform(scaled_shap_values)
            
            # Find the optimal number of clusters
            c = self.find_optimal_clusters(transformed_shap_values, quick=True)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=c, init='k-means++', max_iter=5000, n_init=500, random_state=42)
            clusters = kmeans.fit_predict(transformed_shap_values)
            
            # Generate dendrogram
            scaler = StandardScaler()
            scaled_shap_values = scaler.fit_transform(transformed_shap_values)

            # Create the linkage matrix using a different linkage method if needed
            linkage_matrix = linkage(scaled_shap_values, method='ward')

            # Set a manual color threshold based on the desired number of clusters
            # This uses a heuristic by selecting the threshold just above the last `c-1` merges
            distances = linkage_matrix[:, 2]
            color_threshold = distances[-c] if len(distances) > c else max(distances) / 2

            # Plot the dendrogram
            plt.figure(figsize=(30, 12))
            dendrogram(
                linkage_matrix,
                p=int(c),  # Number of clusters to display
                color_threshold=color_threshold,  # Set the threshold for coloring clusters
                show_leaf_counts=True,
                truncate_mode=None  # Show full dendrogram if possible
            )
            plt.title(f'Hierarchical Clustering Dendrogram for Model {m}')
            plt.xlabel('Sample Index')
            plt.ylabel('Distance')
            plt.savefig(f'dendrogram_model_{m}.png')
            plt.close()
            
            print(f"Dendrogram saved as dendrogram_model_{m}_{self.model}_{self.cohort_name()}.png with {c} clusters.")
        
            
            # Pair each feature name with its cluster label and sensor ID
            feature_cluster_pairs = list(zip(all_feature_names, clusters, all_sensor_ids))
            
            # Populate pred_sensor_kit
            for feature, cluster, sensor_id in feature_cluster_pairs:
                base_feature_name = feature.split('_', 1)[1]  # Remove sensor ID prefix
                if base_feature_name not in pred_sensor_kit[m]:
                    pred_sensor_kit[m][base_feature_name] = []
                pred_sensor_kit[m][base_feature_name].append((sensor_id, cluster))
        return pred_sensor_kit

    def find_optimal_clusters(self, data, max_clusters=10, quick=False):
        """
        Find the optimal number of clusters using the elbow method with silhouette score.
        
        :param data: The data to cluster
        :param max_clusters: The maximum number of clusters to try
        :param quick: If True, use a faster but less thorough method
        :return: The optimal number of clusters
        """
        if quick:
            # Use a quicker method with fewer cluster options
            cluster_range = range(2, min(max_clusters, 6))
        else:
            cluster_range = range(2, max_clusters + 1)
        
        silhouette_scores = []
        
        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=5000, n_init=50, random_state=42)
            cluster_labels = kmeans.fit_predict(data)
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
            
            print(f"For n_clusters = {n_clusters}, the average silhouette score is {silhouette_avg}")
        
        # Find the elbow point
        differences = np.diff(silhouette_scores)
        elbow_point = np.argmax(differences) + 2  # Add 2 because we started with 2 clusters
        self.elbow = elbow_point
        # Plot the elbow curve
        # plt.figure(figsize=(20, 12))
        # plt.plot(cluster_range, silhouette_scores, 'bo-')
        # plt.xlabel('Number of Clusters')
        # plt.ylabel('Average Silhouette Score')
        # plt.title('Elbow Method for Optimal k')
        # plt.vlines(x=elbow_point, ymin=min(silhouette_scores), ymax=max(silhouette_scores), 
        #         colors='r', linestyles='dashed', label=f'Elbow Point: {elbow_point}')
        # plt.legend()
        # plt.show()
        
        return elbow_point

    def show_predictor_scores(self, models, abs_val=False, non_norm=True, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=True):
        pred_kit = {}
        for m in models:
            pred_kit[m] = self.get_predictor_scores_for_model(
                m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm,
            )
        items = {}
        sensor_count = {}
        sensor_accuracy = {}
        all_categories = set()  # Set to store all unique categories

        for model, model_pred_sets in pred_kit.items():
            if first_model_features is not None:
                first_features = model_pred_sets[0][0][0].get_top_n_features(NUM_TOP)
            else:
                first_features = None

            for model_pred_set in model_pred_sets:
                ps, acc_score, sensor_obj = model_pred_set
                sensor_name = sensor_obj.human_name()
                print("AXIS 2", axis)
                shap_scores, top_scores, sensors = self.gen_stats([model_pred_set], abs_val=abs_val, non_norm=non_norm, first_model_features=first_features, num_top=NUM_TOP, use_cat=use_cat, axis=axis, model=model)
                for i, sensor in enumerate(sensors):
                    if sensor not in items:
                        items[sensor] = shap_scores[i]
                        sensor_count[sensor] = 1
                        sensor_accuracy[sensor] = acc_score
                    else:
                        for feature, value in shap_scores[i].items():
                            items[sensor][feature] = items[sensor].get(feature, 0) + value
                        sensor_count[sensor] += 1
                        sensor_accuracy[sensor] += acc_score
                    all_categories.update(shap_scores[i].keys())  # Update the set of all categories

        # Average the accumulated SHAP values and accuracy scores, and append 'ave_accuracy'
        for sensor in items:
            for feature in items[sensor]:
                items[sensor][feature] /= sensor_count[sensor]
            items[sensor]["ave_accuracy"] = sensor_accuracy[sensor] / sensor_count[sensor]

        # Prepare averaged_shap_scores, averaged_top_scores, and averaged_sensors for plotting
        averaged_shap_scores = [{cat: items[sensor].get(cat, 0) for cat in all_categories} for sensor in items]  # Ensure all categories are represented
        averaged_top_scores = averaged_shap_scores
        averaged_sensors = list(items.keys())


        if include_accuracy:
            average_vals = list(items.values())
            i = 0
            while i < len(average_vals):
                averaged_shap_scores[i]['ave_accuracy'] = average_vals[i]['ave_accuracy']
                i += 1
        metric = 'euclidean'
        combined_data, distance_matrix, sensor_ids = self.prepare_sensor_data(metric=metric)
        feature_cluster_map, color_mapping = self.create_dendrogram_with_clusters(combined_data, distance_matrix)
        
        if self.model is None:
            mod = '_'
        else:
            mod = self.model
        
        fcm_title = 'fcm_' + self.cohort_name() + '_' + mod + '_' + metric + '.csv'
        print(fcm_title)

        self.save_feature_cluster_map_to_csv(combined_data, feature_cluster_map, filename=fcm_title)
        # self.plot_shapley_over_time_updated(averaged_shap_scores, feature_cluster_map, color_mapping, averaged_sensors)
        # self.plot_shap_scores_by_sensor_and_cluster(combined_data, feature_cluster_map, color_mapping, sensor_accuracy)
        self.plot_shap_scores_by_sensor_and_cluster(combined_data, feature_cluster_map, color_mapping, sensor_accuracy, normalize=False)
        # return self.new_plot(averaged_shap_scores, averaged_sensors, feature_cluster_map=feature_cluster_map)

    def set_color_mapping(self, feature_cluster_map):
        # Count the number of features in each cluster
        cluster_sizes = collections.Counter(feature_cluster_map.values())
        
        # Sort clusters by size in descending order
        sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
        
        # Initialize color mapping
        color_mapping = {}
        
        # Assign colors based on cluster sizes
        for i, (cluster, _) in enumerate(sorted_clusters):
            if i == 0:
                color_mapping[cluster] = '#ff0000'  # Red for the largest cluster
            elif i == 1:
                color_mapping[cluster] = '#8000ff'  # Purple for next the largest cluster
            else:
                color_mapping[cluster] = '#00ff00'  # Green for any smaller clusters
        
        return color_mapping
        # return self.plot_shap_changes(averaged_shap_scores, averaged_top_scores, averaged_sensors, percentage_of_average=False, first_model_features=None, num_top=NUM_TOP, include_accuracy=include_accuracy, axis=axis)
    def save_feature_cluster_map_to_csv(self, combined_data, feature_cluster_map, filename='shap_data_with_info.csv'):
        """
        Save the SHAP data to a CSV file with additional feature information and cluster.
        
        :param combined_data: pandas DataFrame containing the SHAP values
        :param feature_cluster_map: dict mapping features to their cluster
        :param filename: str, name of the output CSV file
        """
        # Create a list to store feature information
        feature_info = []

        # Iterate through each feature (column) in the DataFrame
        for feature in combined_data.columns:
            params = fn_get_params_for_column(feature)
            cluster = feature_cluster_map.get(feature, 'N/A')
            
            if params:
                feature_info.append({
                    'Feature': feature,
                    'Sensor ID': params['sensor_id'],
                    'Sensor Code': params['sensor_code'],
                    'Feature Name': params['feature_name'],
                    'Feature Index': params['feature_index'],
                    'Axis': params['axis'],
                    'Parameters': str(params['params']),  # Convert to string for CSV compatibility
                    'Cluster': cluster
                })
            else:
                feature_info.append({
                    'Feature': feature,
                    'Sensor ID': 'N/A',
                    'Sensor Code': 'N/A',
                    'Feature Name': 'N/A',
                    'Feature Index': 'N/A',
                    'Axis': 'N/A',
                    'Parameters': 'N/A',
                    'Cluster': cluster
                })

        # Create a DataFrame with feature information
        feature_info_df = pd.DataFrame(feature_info)

        # Transpose the original data for better CSV format
        transposed_data = combined_data.T

        # Merge feature information with transposed data
        result = pd.merge(feature_info_df, transposed_data, left_on='Feature', right_index=True)
        # Prepare filename
        filename = self.cohort_name() + filename
        if self.model is not None:
            filename = self.model + '_' + filename

        # Save to CSV
        result.to_csv(filename, index=False)
        print(f"CSV file '{filename}' has been created successfully.")


    def plot_shap_scores_by_sensor_and_cluster(self, df, feature_cluster_map, color_mapping, sensor_accuracy, normalize=True):
        print(sensor_accuracy)
        # Initialize a dictionary to store mean SHAP values by sensor and cluster
        sensor_cluster_means = {sensor: {cluster: [] for cluster in color_mapping.keys()} for sensor in sensor_accuracy.keys()}
        
        # Iterate through the columns and assign SHAP values to the appropriate sensor and cluster
        for col in df.columns:
            sensor_id = int(col.split('_')[0])
            sensor_name = Sensor.get(sensor_id).human_name()  # Replace with the correct method to get sensor names
            cluster = feature_cluster_map.get(col)
            if sensor_name in sensor_cluster_means and cluster is not None:
                sensor_cluster_means[sensor_name][cluster].append(df[col])
            else:
                print("NOT FOUND", sensor_name, cluster)
        
        # Correct aggregation: Calculate mean SHAP values per sensor and cluster
        for sensor in sensor_cluster_means:
            for cluster in sensor_cluster_means[sensor]:
                if sensor_cluster_means[sensor][cluster]:
                    # Aggregate correctly by taking the mean of each feature per cluster
                    sensor_cluster_means[sensor][cluster] = pd.concat(sensor_cluster_means[sensor][cluster], axis=1).mean(axis=1)
                else:
                    # If no data for a cluster, fill with NaN
                    sensor_cluster_means[sensor][cluster] = pd.Series([np.nan] * df.shape[0])
        
        # Optional normalization of SHAP values
        if normalize:
            for sensor in sensor_cluster_means:
                for cluster in sensor_cluster_means[sensor]:
                    max_val = sensor_cluster_means[sensor][cluster].abs().max()
                    if max_val != 0:
                        sensor_cluster_means[sensor][cluster] /= max_val
        
        # Plotting
        fig, ax1 = plt.subplots(figsize=(14, 10))

        # Plot SHAP values
        first_label_shown = {cluster: False for cluster in color_mapping.keys()}  # Track if cluster label has been shown
        for sensor in sensor_accuracy.keys():
            for cluster, color in color_mapping.items():
                y = sensor_cluster_means[sensor][cluster]
                x = [sensor] * len(y)  # X-axis values (sensor name repeated)
                # Filter out NaN values to avoid plotting issues
                mask = ~y.isna()
                if mask.sum() == 0:  # Skip plotting if all values are NaN
                    continue
                ax1.scatter(
                    np.array(x)[mask],
                    y[mask],
                    color=color,
                    label=f'Cluster {cluster}' if not first_label_shown[cluster] else "",
                    alpha=0.3,  # Increased transparency
                    edgecolors='none'  # Remove edge colors for better overlap visibility
                )
                first_label_shown[cluster] = True  # Mark the label as shown

        # Plot accuracy scores on the same axes
        ax1_2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis
        sensor_positions = list(sensor_accuracy.keys())
        accuracy_scores = list(sensor_accuracy.values())
        import pdb;pdb.set_trace()
        ax1_2.plot(sensor_positions, accuracy_scores, 'o-', color='green', label='Accuracy Score', markersize=10, alpha=0.7)
        ax1_2.set_ylim(0, 1)  # Assuming accuracy scores are between 0 and 1


        # Adjust labels, titles, and legends
        if normalize:
            is_norm = "Normalized"
        else:
            is_norm = ""

        if self.cohort_name() == 'cp_before':
            cohort = "CP Patient"
        else:
            cohort = "Healthy Control"

        if self.model == "default" or self.model == None:
            mod = "Sub Motion"
        else:
            mod = "Full Motion"

        title = f"{is_norm} {cohort} SHAP Value Clusters by Sensor--{mod}"
        ax1.set_title(title)
        ax1.set_xlabel('Sensor Position')
        ax1.set_ylabel('Average SHAP Value')


        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.grid(True)
        ax1.legend(title="Clusters", loc='upper left')

        ax1_2.set_ylabel('Accuracy Score')
        ax1_2.legend(loc='upper right')
        if not normalize:
            ax1.set_ylim(-.4, .4)

        # Adjust layout and save the figure before showing it
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the title
        plt.savefig(title, dpi=300, bbox_inches='tight')
        plt.show()


    def get_new_axis(self, model=None, abs_val=False, non_norm=True):
        cluster_assignments, unique_colors = self.linkage_mat(view=False, abs_val=abs_val, non_norm=non_norm, model=model, save=True)

        self.unique_colors = unique_colors
        
        reversed_assignments = {}
        for feature, cluster in cluster_assignments.items():
            if cluster not in reversed_assignments:
                reversed_assignments[cluster] = []
            reversed_assignments[cluster].append(feature)
        return reversed_assignments

    def gen_stats(self, model_pred_set, abs_val=False, non_norm=True, first_model_features=None, num_top=NUM_TOP, use_cat=False, axis=False, model=None):
        shap_scores = []
        top_scores = []
        sensors = []
        predictor_score = model_pred_set[0][0]

        if first_model_features is True:
            top_features = predictor_score[0].get_top_n_features(num_top)
            first_shap_scores = current_model_pred_set[0].get_shap_values_for_features(top_features)

        for current_model_pred_set, acc_score, snr in model_pred_set:
            if current_model_pred_set == []:
                print("PredictorScore not found!")
                continue

            if first_model_features is True:
                current_top_features = top_features
                current_shap_scores = first_shap_scores
            else:
                current_top_features = current_model_pred_set[0].get_top_n_features(num_top)
                current_shap_scores = current_model_pred_set[0].get_shap_values_for_features(current_top_features)
            
            # if use_cat is True:
            #     if axis is True:
            #         if self.curr_axis is None or self.curr_axis is (None,):
            #             axis = self.get_new_axis(model=model, abs_val=abs_val, non_norm=non_norm)
            #             self.curr_axis = axis
            #         else:
            #             axis = self.curr_axis
                
            #     print(axis)
            #     print(current_shap_scores, axis)
            #     aggregate_shap_values = current_model_pred_set[0].aggregate_shap_values_by_category(current_shap_scores, axis)
            #     normalized_shap_values = current_model_pred_set[0].normalize_shap_values(aggregate_shap_values)
            #     # current_shap_scores = normalized_shap_values

            shap_scores.append(current_shap_scores)
            top_scores.append(current_shap_scores)
            sensors.append(snr.human_name())
        return [shap_scores, top_scores, sensors]
    
    import numpy as np
    from scipy.cluster.hierarchy import linkage, fcluster
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    # def generate_color_mapping(self, shap_scores):
    #     """
    #     Generate a color mapping for features based on SHAP scores.
        
    #     Parameters:
    #     shap_scores (list of dict): List of dictionaries containing SHAP scores for each feature
        
    #     Returns:
    #     dict: Mapping of features to their assigned colors
    #     """
    #     # Extract unique features (excluding 'ave_accuracy')
    #     unique_features = list(set(key for score in shap_scores for key in score.keys() if key != 'ave_accuracy'))
        
    #     # Create a similarity matrix based on SHAP score patterns
    #     sim_matrix = np.zeros((len(unique_features), len(unique_features)))
    #     for i, feature1 in enumerate(unique_features):
    #         for j, feature2 in enumerate(unique_features):
    #             if i != j:
    #                 corr = np.corrcoef([score.get(feature1, 0) for score in shap_scores],
    #                                 [score.get(feature2, 0) for score in shap_scores])[0, 1]
    #                 sim_matrix[i, j] = sim_matrix[j, i] = abs(corr)  # Use absolute correlation as similarity
        
    #     # Generate the linkage matrix
    #     Z = linkage(1 - sim_matrix, 'ward')  # Use 1-similarity as distance
        
    #     # Find the optimal color threshold based on the largest difference in heights
    #     heights = Z[:, 2]
    #     diffs = np.diff(heights)
    #     color_threshold = heights[np.argmax(diffs)]
        
    #     # Perform the clustering
    #     cluster_labels = fcluster(Z, color_threshold, criterion='distance')
        
    #     # Generate a color for each unique cluster
    #     unique_clusters = np.unique(cluster_labels)
    #     color_palette = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
    #     cluster_colors = {cluster: mcolors.to_hex(color) 
    #                     for cluster, color in zip(unique_clusters, color_palette)}
        
    #     # Create the feature to color mapping
    #     unique_colors = {feature: cluster_colors[cluster] 
    #                     for feature, cluster in zip(unique_features, cluster_labels)}
        
    #     return unique_colors



    # def plot_shap_changes(self, shap_scores, top_scores, sensors, percentage_of_average=False, first_model_features=None, num_top=NUM_TOP, include_accuracy=False, axis=False):
    #     plt.figure(figsize=(12, 8))
        
    #     self.unique_colors = self.generate_color_mapping(shap_scores)
    #     import pdb;pdb.set_trace()
    #     for feature, color in self.unique_colors.items():
    #         print("FEATURE", feature, "COLOR", color)
    #         shap_values = [score.get(feature, 0) for score in shap_scores]
    #         if percentage_of_average is True:
    #             averages = [sum(score.values()) / len(score) for score in shap_scores]
    #             shap_values = [100 * val / avg if avg != 0 else 0 for val, avg in zip(shap_values, averages)]
            
    #         plt.plot(sensors, shap_values, label=feature, color=color, marker='o', linestyle='-')

    #     # Optionally plot 'ave_accuracy' if requested
    #     import pdb;pdb.set_trace()
    #     if include_accuracy is True:
    #         accuracy_values = [score.get("ave_accuracy", 0) for score in shap_scores]
    #         plt.plot(sensors, accuracy_values, label='ave_accuracy', color='red', marker='x', linestyle='--', linewidth=2)

    #     if axis:
    #         category = "Axis"
    #     else:
    #         category = "Feature"

    #     desc = Task.get(self.task_id).description.split('_')
    
    #     task_title = category + " " + desc[0]
    #     plt.title(task_title + " Categories' SHAP Values Across Sensors" + (" (Average Accuracy) " if include_accuracy else ""))
    #     plt.xlabel("Sensor Location")
    #     plt.ylabel("SHAP Value (Average Accuracy)" + (" (% of Average)" if percentage_of_average else ""))
    #     # plt.xticks(rotation=45)
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    #     plt.ylim(0, 1.4)
    #     plt.tight_layout()
    #     plt.grid(True)
    #     plt.show()

    def get_shap_values_from_scores(self, model_name):
        ps = self.get_predictor_scores_for_model(model_name)

    def get_shap_values_for_model(self, model_name):
        self.get_predictor_scores_for_model
    
    def load_items(self):
        with open('items.pickle', 'rb') as handle:
            self.items = pickle.load(handle)
    
    def assemble_bdf(self, non_norm=True, abs_val=False, force_load=False, skip_default_sensors=False):
        prs = Predictor.where(task_id=self.task_id, multi_predictor_id=self.id, cohort_id=self.cohort_id)
        for el in prs:
            el.update(accuracies=None)
        
        self.gen_items_for_sensors()


    def gen_scores_for_sensor(self, non_norm=True, abs_val=False, force_load=False, skip_default_sensors=False, add_other=False, rand_skip=False, abs_skip=True):
        preds = Predictor.where(task_id=self.task_id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
        print(len(preds), " existing predictors found.")

        if skip_default_sensors is True:
            curr_sensors = self.sensors
        else:
            curr_sensors = self.get_sensors()

        time_threshold = datetime.now() - timedelta(hours=36)

        for sensor in curr_sensors:
            if sensor.name != "relbm_x":
                predictor = Predictor.find_or_create(task_id=self.task_id, sensor_id=sensor.id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
                if predictor.accuracies is None or force_load is True:
                    predictor = predictor.train_from(force_load=True, add_other=add_other)
                else:
                    print("Skipping for now because acc or bad id", predictor.accuracies)
                    
        with open('items.pickle', 'wb') as handle:
            pickle.dump(self.items, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done training Predictors for MP:", self.id)
        
    def get_all_preds(self, same_cohort=False):
        if same_cohort:
            return Predictor.where(multi_predictor_id=self.id, cohort_id=self.cohort_id)
        else:
            return Predictor.where(multi_predictor_id=self.id)
            

    def get_norm_acc(self, alt=False):
        pr = self.get_norm_predictors()
        return self.get_acc(preds=pr, alt=alt)
    
    def get_abs_acc(self, alt=False):
        return self.get_acc(abs_val=True, alt=alt)
    
    def compare_acc(self, aa=None, bb=None):
        # Convert the lists to dictionaries for easier access
        if aa == None:
            aa = self.get_all_acc(alt="training")
            
        if bb == None:
            bb = self.get_all_acc(alt=True)
            
        aa_dict = {sensor: scores for sensor, scores in aa}
        bb_dict = {sensor: scores for sensor, scores in bb}

        # Initialize a list to hold the differences
        difference_list = []

        # Iterate over the keys (sensor names) in the test data (aa)
        for sensor, aa_scores in aa_dict.items():
            # Check if the sensor is also in the training data (bb)
            if sensor in bb_dict:
                bb_scores = bb_dict[sensor]
                
                # Calculate differences for each model in this sensor
                model_differences = {model: aa_scores.get(model, 0) - bb_scores.get(model, 0) for model in set(aa_scores).union(bb_scores)}
                
                # Append the sensor and its differences to the list
                difference_list.append((sensor, model_differences))

        # Output the difference_list which contains the differences in the same format as bb
        return difference_list


    
    def get_all_acc(self, alt=False):
        pr = self.get_all_preds()
        return self.get_acc(preds=pr, alt=alt)
    
    def get_acc(self, non_norm=True, abs_val=False, all=False, preds=None, alt=None):
        if all is True:
            predictors = self.get_all_preds()
        elif preds is not None:
            predictors = preds
        else:
            predictors = self.get_predictors(non_norm=non_norm, abs_val=abs_val)

        results = []
        # Iterate through the mpss list
        for pr in predictors:
            # Extracting the classifier accuracies
            if pr.get_accuracies() != {} and pr.get_accuracies() != None:
                if alt is None:
                    accuracies = pr.get_accuracies()['classifier_accuracies']
                else:
                    accuracies = pr.get_classifier_accuracies(alt=alt)
            else:
                print("Accuracies not found:", pr.attrs())
                continue
            
            # Sorting by accuracy in descending order and rounding to three decimal places
            sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}

            print(sorted_accuracies)
            
            # Append sensor name and its sorted accuracies to the results list
            results.append((pr.sensor().name, sorted_accuracies))
        results.sort(key=lambda x: max(x[1].values()), reverse=True)

        return results
    
    def get_norm_corr(self, non_norm=True, abs_val=False):
        predictors = self.get_norm_preds()
        prs = []
        for pr in predictors:
            prs.append((pr.sensor().name, pr.get_fold_corr()))
        return prs

    
    def cohort_name(self):
        return Cohort.get(id=self.cohort_id).name

    def gen_scores_for_mp(self, force_load=False):
        # self.view_progress(fix=False, multi=True)
        # self.gen_scores_for_sensor()
        print("done with default")

        self.gen_scores_for_sensor(force_load=force_load)
        self.gen_scores_for_sensor(abs_val=True, force_load=force_load)
        self.gen_scores_for_sensor(non_norm=False, force_load=force_load)

        norm_pred =  self.get_norm_predictors()

        #TODO REMOVE THIS STEPHEN
        if self.model == "grad_set":
            self.gen_train_combo_mp(use_norm_pred=True, norm_pred=norm_pred, model="grad_set_combo")
        else:
            self.gen_train_combo_mp(use_norm_pred=True, norm_pred=norm_pred)

        print("really done")
    
    
    def view_progress(self, fix_missing_acc=False, multi=False):
        if multi:
            mp = MultiPredictor.where(cohort_id=2, model='norm_non_abs_combo')[0]
            # all_preds = mp.get_all_preds(same_cohort=True)
            all_preds = Predictor.where(multi_predictor_id=mp.id)
        else:
            all_preds = self.get_all_preds(same_cohort=True)
        
        print("COHORT", Cohort.get(self.cohort_id).name)
        print(self.attrs())
        print("TOTAL number of predictors", len(all_preds))
        
        default = []
        abs_val = []
        norm = []
        others = []
        
        for pred in all_preds:
            if pred.non_norm and not pred.abs_val:
                default.append(pred)
            elif pred.abs_val and pred.non_norm:
                abs_val.append(pred)
            elif not pred.non_norm:
                norm.append(pred)
            else:
                print("OTHER", pred.attrs())
        
        try:
            all_pred_types = list(reversed([default, abs_val, norm, others]))  # Convert reversed iterator to list
            for pred_type in all_pred_types:
                print("\nNEW PRED TYPE\n")
                if not pred_type:
                    print("Empty pred_type, skipping.")
                    continue
                for pred in pred_type:
                    sensor = pred.sensor()
                    acc = pred.get_classifier_accuracies()
                    print(f"Classifier accuracies for ID {pred.id}, SENSOR {sensor.name}, SENSOR ID {sensor.id}: {acc}, updated: {pred.updated_at}, non_norm {pred.non_norm}, abs {pred.abs_val}")

                    if fix_missing_acc and acc == None:
                        print("FIXING")
                        print("    ID:", pred.id,"SENSOR:", sensor.name, "ACCURACIES:", acc, "UPDATED AT", pred.updated_at)
                        pred.train_from(force_load=True)
                print("\nEND PRED TYPE\n")
        
            print("DONEEEE")
        
        except Exception as e:
            print("An error occurred:", e)
            import traceback
            traceback.print_exc()

# Add the line that starts the pdb debugger again, if needed:
# import pdb; pdb.set_trace()

        


    def view_result_heatmap(self):
        results = self.get_acc()
        print(results)

        return MatrixPlotter.view_and_save_results(results, task="Reg")
    
    def save_abs_shap(self, abs_val=True, non_norm=True):
        preds = self.get_abs_preds()
        if len(preds) is 0:
            print("None!")
        return self.save_shap_values(abs_val=abs_val, non_norm=non_norm, title="abs", preds=None)
    
    def save_norm_shap(self, abs_val=True, non_norm=False):
        preds = self.get_norm_preds()
        if len(preds) is 0:
            print("None!")

        return self.save_shap_values(abs_val=abs_val, non_norm=non_norm, title="norm", preds=preds)
    
    def save_shap(self, abs_val=False, non_norm=True):
        preds = self.get_preds()
        if len(preds) is 0:
            print("None!")
        return self.save_shap_values(abs_val=abs_val, non_norm=non_norm, title="default", preds=preds)
    
    def save_combo_shap(self, abs_val=False, non_norm=False):
        preds = self.get_all_preds()

        return self.save_shap_values(abs_val=abs_val, non_norm=True, title="combo", preds=preds)

    def save_shap_values(self, abs_val=False, non_norm=True, title=None, preds=None):
        if preds is None:
            pr = self.get_predictors(abs_val=abs_val, non_norm=non_norm)
        else:
            pr = preds

        #TODO stephen, change this back
        # title = self.cohort_name() + "/" + title
        if title is not None:
            title = "CP5" + "/" + title
        else:
            title = "CP5" + "/"

        
        for pred in pr:
            scores = pred.get_predictor_scores()
            for score in scores:
                if self.cohort_name() != "healthy_controls":
                    print("yoolo")
                    print(self.cohort_name())
                    score.view_shap_plot(title=title, abs_val=abs_val, non_norm=non_norm)
                elif title != None:
                    print(title)
                    print(self.model)
                    # if self.model == "norm_non_abs_combo":
                    #     title = (title + "/" + "combination")
                    score.view_shap_plot(title=title, abs_val=abs_val, non_norm=non_norm)
                else:
                    print("yo")
                    score.view_shap_plot()

    
    def gen_train_combo_mp(self, model="non_abs_combo", use_norm_pred=False, norm_pred=None, get_sg_count=False, custom_mp=None, force_load=False):
        default_predictors = self.get_predictors()
        abs_predictors = self.get_abs_predictors()

        if use_norm_pred and norm_pred is not None:
            norm_predictors = norm_pred 
        elif use_norm_pred:
            print("using norm preds")
            norm_predictors = self.get_norm_predictors() 

        norm_dict = {pred.sensor_id: pred for pred in norm_predictors} if norm_predictors else {}
        keep_old = model == "non_abs_combo"

        if custom_mp is not None:
            new_mp = custom_mp
        else:
            new_mp = MultiPredictor.find_or_create(
                cohort_id=self.cohort_id, task_id=self.task_id, model=("norm_" + "non_abs_combo") if keep_old else model
            )
        default_dict = {pred.sensor_id: pred for pred in default_predictors}
        abs_dict = {pred.sensor_id: pred for pred in abs_predictors}
        
        combos = {}
        for sensor_id, default_pred in default_dict.items():
            if sensor_id in abs_dict:
                combos[sensor_id] = [default_pred, abs_dict[sensor_id], norm_dict[sensor_id]]

        for combo_sensor_id, preds in reversed(combos.items()):
            print(preds)
            print("Training for sensor:", Sensor.get(combo_sensor_id).name)
            new_pred = Predictor.find_or_create(task_id=self.task_id, sensor_id=combo_sensor_id, multi_predictor_id=new_mp.id, cohort_id=self.cohort_id)
            #TODO Stephen remove this after finishing cp trials
            
            if norm_predictors:
                new_mp.combo_train(preds[0], preds[1], new_pred, norm_pred=preds[2], get_sg_count=get_sg_count, force_load=force_load)
            else:
                new_mp.combo_train(preds[0], preds[1], new_pred, get_sg_count=get_sg_count, force_load=force_load)

        print("Done!")


    def get_combo_df(self):
        return pickle.loads(self.normalized)


    def combo_train(self, default_pred, abs_pred, new_pred, norm_pred=None, classifier_name=None, remove_lateral=True, force_load=False, get_sg_count=False):
        # non abs non norm
        
        result = default_pred.get_final_bdf(untrimmed=True, force_abs_x=remove_lateral, get_sg_count=get_sg_count, force_load=force_load)


        if result is None or (isinstance(result[0], pd.DataFrame) and result[0].empty):
            print("EMPTY DF SKIPPING")
            return None
        df1, y1 = result
        # abs non norm
        result = abs_pred.get_final_bdf(untrimmed=True, force_abs_x=False, get_sg_count=get_sg_count, force_load=force_load)
        if result is None or (isinstance(result[0], pd.DataFrame) and result[0].empty):
            print("EMPTY DF SKIPPING")
            return None
        df2, _ = result

        # norm (abs)
        df3, _ = norm_pred.get_final_bdf(untrimmed=True, force_abs_x=False, get_sg_count=get_sg_count, force_load=force_load)
        if result is None or (isinstance(result[0], pd.DataFrame) and result[0].empty):
            print("EMPTY DF SKIPPING")
            return None
        df3, _ = result


        comp_df = MultiPredictor.create_composite_dataframe(df1, df2, df3=df3)
        combo_df = Predictor.trim_bdf_with_boruta(comp_df, y1)

        pickled_df = pickle.dumps(combo_df)

        new_pred.update(aggregated_stats=pickled_df)
        if len(combo_df.columns) <= 2:
            combo_df = new_pred.trim_bdf(comp_df, custom_limit=48)
        
        classifiers = Predictor.define_classifiers_cls(False, False, classifier_name=classifier_name, feature_count=len(combo_df.columns))
        groups = combo_df['patient']
        
        y = new_pred.get_y(combo_df)

        print("current combo", combo_df)
        classifier_accuracies, classifier_params, classifier_metrics = {}, {}, {}
        scores = {}
        params = {}
        acc_metrics = {}
        for classifier_name, classifier_data in classifiers.items():
            print(f"Training {classifier_name}...")
            classifier_scores, best_params, extra_metrics, scores, params, acc_metrics = Predictor._train_from_mp(combo_df, y, groups, classifier_name, classifier_data, scores, params, acc_metrics, mps=self, use_shap=True, new_pred=new_pred)
            classifier_accuracies[classifier_name] = np.mean(classifier_scores)
            classifier_params[classifier_name] = best_params
            classifier_metrics[classifier_name] = extra_metrics

        print("Accuracy Scores:")
        for classifier_name, score in scores.items():
            print(f"{classifier_name}: {score}")
            classifier_accuracies[classifier_name] = score  # Store the scores in the new dictionary
        
        for classifier_name, param in params.items():
            print(f"{classifier_name}: {params}")
            classifier_params[classifier_name] = param  # Store the scores in the new dictionary
        
        for classifier_name, n_metrics in acc_metrics.items():
            print(f"{classifier_name}: {n_metrics}")
            classifier_metrics[classifier_name] = n_metrics 
        
        new_pred._update_accuracies(classifier_accuracies, classifier_params, classifier_metrics)
        
        new_pred.save()
        print("done......")

    @classmethod
    def create_composite_dataframe(cls, df1, df2, df3=None):
        # [previous code for type conversion]

        # Merge df1 and df2
        merged_df = pd.merge(df1, df2, on=['patient', 'is_dominant'], suffixes=('_default', '_abs'))

        # If df3 is to be used, merge it as well
        if df3 is not None:
            df3['patient'] = df3['patient'].astype(int)
            df3['is_dominant'] = df3['is_dominant'].astype(int)
            # Use 'suffixes' to handle overlapping column names
            
            new_df = pd.merge(merged_df, df3, on=['patient', 'is_dominant'], suffixes=('', '_norm'))
            return new_df

        return merged_df

    @classmethod
    def make_averages_csv(cls, mpss, model_name):

        # Loop through the sensors and accumulate the metrics
        # Initialize a dictionary to hold the sums of all metrics for each model
        metrics_sum = {}
        metrics_count = {}

        # Loop through the sensors and accumulate the metrics
        for mp in mpss:
            accuracies = mp.get_accuracies()['classifier_accuracies']
            metrics = mp.get_accuracies()['classifier_metrics']
            for model, accuracy in accuracies.items():
                if model not in metrics_sum:
                    metrics_sum[model] = {
                        'Accuracy': 0,
                        'AUC-ROC': 0,
                        'F1-score': 0,
                        'Log loss': 0,
                        'Precision': 0,
                        'Recall': 0,
                    }
                    metrics_count[model] = 0
                
                metrics_sum[model]['Accuracy'] += accuracy
                metrics_sum[model]['AUC-ROC'] += metrics[model]['AUC-ROC']
                metrics_sum[model]['F1-score'] += metrics[model]['F1-score']
                metrics_sum[model]['Log loss'] += metrics[model]['Log loss']
                metrics_sum[model]['Precision'] += metrics[model]['Precision:']
                metrics_sum[model]['Recall'] += metrics[model]['Recall:']
                
                metrics_count[model] += 1

        # Compute the averages
        metrics_avg = {model: {metric: total / metrics_count[model] for metric, total in metrics.items()} for model, metrics in metrics_sum.items()}

        # Write the averages to a CSV file
        file_name = model_name + '_averages.csv'
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Accuracy', 'AUC-ROC', 'F1-score', 'Log loss', 'Precision', 'Recall'])
            for model, metrics in metrics_avg.items():
                writer.writerow([model, metrics['Accuracy'], metrics['AUC-ROC'], metrics['F1-score'], metrics['Log loss'], metrics['Precision'], metrics['Recall']])


        
    # def cluster_features_shap(self, sensor_shap_df):
    #     # Standardize the features
    #     scaler = StandardScaler()
    #     standardized_shap_values = scaler.fit_transform(sensor_shap_df)
        
    #     # Use the Elbow Method to find the optimal number of clusters
    #     wcss = []
    #     for i in range(1, 10):
    #         kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #         kmeans.fit(standardized_shap_values.T)  # Transpose to cluster features instead of samples
    #         wcss.append(kmeans.inertia_)
        
    #     # Assuming you determined the optimal number of clusters (this could be automated or manually reviewed)
    #     optimal_clusters = 3  # Example value
        
    #     kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     clusters = kmeans.fit_predict(standardized_shap_values.T)  # Again, transpose so features are data points
        
    #     return clusters

    # TODO
    # 1. update plot colors, verify clusters
    # 2. add methods
    # 3. 