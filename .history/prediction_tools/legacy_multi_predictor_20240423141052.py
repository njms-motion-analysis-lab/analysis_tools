# import sqlite3

import csv
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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from prediction_tools.predictor_score import PredictorScore
from viewers.matrix_plotter import MatrixPlotter
from viewers.multi_plotter import MultiPlotter


SENSOR_CODES = [
    'rfin_x',
    'rwra_x',
    # 'rwrb_x',
    'rfrm_x',
    'relb_x',
    # 'relbm_x',
    'rupa_x',
    'rsho_x',
    # 'rbhd_x',
    # 'rfhd_x',
]

FEATURE_COLORS = {
    'C0': '#1f77b4',  # muted blue
    'C1': '#ff7f0e',   # safety orange
    # 'autocorrelation': '#2ca02c',  # cooked asparagus green
    # 'trend_based': '#d62728',  # brick red
    'C2': '#9467bd',  # muted purple
    # 'wavelet_based': '#8c564b',  # chestnut brown
    # 'complexity_measures': '#e377c2',  # raspberry yogurt pink
    'C3': '#7f7f7f',  # middle gray
    'C4': '#bcbd22',  # curry yellow-green
    'C5': '#17becf'  # blue-teal
    # 'y-axis' :'green',
    # 'x-axis' : 'red',
    # 'z-axis' : 'blue',
}
# TOP_TREE_MODELS = ['RandomForest', 'ExtraTrees', 'XGBoost']
TOP_TREE_MODELS = ['RandomForest', 'ExtraTrees', 'CatBoost']
DISTANCE_THRESHOLD = 20

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
        self.abs_val = False
        

    def gen_items_for_sensors(self, sensors=None, ntaf=False):
        if sensors is None:
            sensors = self.sensors

        for sen in sensors:
            self.gen_items_for_sensor(sen, ntaf)

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
        print("PRED LEN 1", len(preds))
        combos = []
        for pr in preds:
            print("score_type", len(pr.get_predictor_scores()))
            if len(pr.get_predictor_scores()) == 0:
                continue
            print(pr.get_accuracies() != {})
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
        print("yolo")
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

    def show_norm_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
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

    def show_combo_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
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

    def show_all_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
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
    
    def show_default_scores(self, models=TOP_TREE_MODELS, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
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
    

    
    def linkage_mat(self, sim_matrix=None, view=True, abs_val=False, non_norm=True):
        if sim_matrix is None:
            sim_matrix, unique_features = self.sim_matrix(abs_val=abs_val, non_norm=non_norm)
        # Generate the linkage matrix
        Z = linkage(1 - sim_matrix, 'ward')  # Use 1-similarity as distance

        # Assuming 'Z' is your linkage matrix calculated from 'linkage'
        # and 'max_d' is your maximum distance for clusters

        # Create the dendrogram
        dendro = dendrogram(Z, labels=unique_features, color_threshold=DISTANCE_THRESHOLD)

        # Create a legend for the colors. 'dendro['color_list']' holds the cluster colors.
        # We need to remove duplicates and sort the colors to create a meaningful legend.

        legend_handles = []
        for color in sorted(set(dendro['color_list'])):
            if color != 'b':  # Exclude the color if it's the default blue used for the last p merges
                legend_handles.append(plt.Line2D([0], [0], color=color, lw=4))

        plt.legend(legend_handles, [f"Cluster {i+1}" for i in range(len(legend_handles))], loc='upper right')

        if view is True:
            plt.show()
        else:
            # Extract the color information
            leaf_colors = dendro['color_list']
            leaf_labels = dendro['ivl']

            # Map from labels to colors
            label_color_map = {label: color for label, color in zip(leaf_labels, leaf_colors)}
            
            return label_color_map


    def sim_matrix(self, feature_cluster_map=None, non_norm=False, abs_val=False):
        print("ABS VAL", abs_val, "non_norm", non_norm)
        if feature_cluster_map is None:
            feature_cluster_map = self.feature_cluster_map(non_norm=non_norm, abs_val=abs_val)

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

    def feature_cluster_map(self, non_norm=True, abs_val=False):
        sensor_output = self.aggregate_shap_values(non_norm=non_norm, abs_val=abs_val)
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
            pred_kit[m] = self.get_predictor_scores_for_model(
                m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm,
            )
            # Ensure pred_sensor_kit[m] is initialized as a dictionary for the model
            pred_sensor_kit[m] = {}
            for predictor_score in pred_kit[m]:
                # Perform clustering for each sensor and model
                cluster_assignments = predictor_score[0][0].cluster_features_shap()
                # Assuming sensor_object has a unique identifier, such as an ID or a name
                sensor_id = predictor_score[-1].id
                # Adjust the structure of pred_sensor_kit to include sensor information
                for feature, cluster in cluster_assignments:
                    if feature not in pred_sensor_kit[m]:
                        pred_sensor_kit[m][feature] = []
                    pred_sensor_kit[m][feature].append((sensor_id, cluster))
        return pred_sensor_kit

        
    def show_predictor_scores(self, models, abs_val=False, non_norm=True, reverse_order=True, first_model_features=None, num_top=NUM_TOP, use_cat=True, axis=False, include_accuracy=False):
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
                shap_scores, top_scores, sensors = self.gen_stats([model_pred_set], abs_val=abs_val, non_norm=non_norm, first_model_features=first_features, num_top=NUM_TOP, use_cat=use_cat, axis=axis)
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

        return self.plot_shap_changes(averaged_shap_scores, averaged_top_scores, averaged_sensors, percentage_of_average=False, first_model_features=None, num_top=NUM_TOP, include_accuracy=include_accuracy, axis=axis)

    def get_new_axis(self, abs_val=False, non_norm=True):
        cluster_assignments = self.linkage_mat(view=False, abs_val=abs_val, non_norm=non_norm)
        reversed_assignments = {}
        for feature, cluster in cluster_assignments.items():
            if cluster not in reversed_assignments:
                reversed_assignments[cluster] = []
            reversed_assignments[cluster].append(feature)
        return reversed_assignments

    def gen_stats(self, model_pred_set, abs_val=False, non_norm=True, first_model_features=None, num_top=NUM_TOP, use_cat=False, axis=False):
        shap_scores = []
        top_scores = []
        sensors = []
        predictor_score = model_pred_set[0][0]
        
        if first_model_features is True:
            top_features = predictor_score[0].get_top_n_features(NUM_TOP)
            first_shap_scores = current_model_pred_set[0].get_shap_values_for_features(top_features)

        for current_model_pred_set, acc_score, snr in model_pred_set:
            if current_model_pred_set == []:
                print("PredictorScore not found!")
                continue
            
            if first_model_features is True:
                current_top_features = top_features
                current_shap_scores = first_shap_scores
            else:
                current_top_features = current_model_pred_set[0].get_top_n_features(NUM_TOP)
                current_shap_scores = current_model_pred_set[0].get_shap_values_for_features(current_top_features)

            if use_cat is True:
                if axis is True:
                    axis = self.get_new_axis(abs_val=abs_val, non_norm=non_norm)
                print(axis)
                # import pdb;pdb.set_trace()

                aggregate_shap_values = current_model_pred_set[0].aggregate_shap_values_by_category(current_shap_scores, axis)
                normalized_shap_values = current_model_pred_set[0].normalize_shap_values(aggregate_shap_values)
                # normalized_shap_values = aggregate_shap_values
                
                current_shap_scores = normalized_shap_values

            shap_scores.append(current_shap_scores)
            top_scores.append(current_shap_scores)
            sensors.append(snr.human_name())
        return [shap_scores, top_scores, sensors]


    def plot_shap_changes(self, shap_scores, top_scores, sensors, percentage_of_average=False, first_model_features=None, num_top=NUM_TOP, include_accuracy=False, axis=False):
        # Gather all unique feature categories from the shap_scores
        unique_features = set().union(*[score.keys() for score in shap_scores if score])
        if include_accuracy and "ave_accuracy" in unique_features:
            unique_features.remove("ave_accuracy")  # Remove 'ave_accuracy' from plotting categories if included separately
        
        color_map = plt.cm.get_cmap('viridis', len(unique_features) + (1 if include_accuracy else 0))
        feature_to_color = {feature: color_map(i) for i, feature in enumerate(unique_features)}

        plt.figure(figsize=(12, 8))
        
        for feature, color in FEATURE_COLORS.items():
            shap_values = [score.get(feature, 0) for score in shap_scores]
            if percentage_of_average:
                averages = [sum(score.values()) / len(score) for score in shap_scores]
                shap_values = [100 * val / avg if avg != 0 else 0 for val, avg in zip(shap_values, averages)]
            
            plt.plot(sensors, shap_values, label=feature, color=color, marker='o', linestyle='-')

        # Optionally plot 'ave_accuracy' if requested
        if include_accuracy:
            accuracy_values = [score.get("ave_accuracy", 0) for score in shap_scores]
            plt.plot(sensors, accuracy_values, label='ave_accuracy', color='red', marker='x', linestyle='--', linewidth=2)

        if axis:
            category = "Axis"
        else:
            category = "Feature"
        
        desc = Task.get(self.task_id).description.split('_')
    
        task_title = category + " " + desc[0]
        plt.title(task_title + " Categories' SHAP Values Across Sensors" + (" (Average Accuracy) " if include_accuracy else ""))
        plt.xlabel("Sensor Location")
        plt.ylabel("SHAP Value (Average Accuracy)" + (" (% of Average)" if percentage_of_average else ""))
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.ylim(0, 1.4)
        plt.tight_layout()
        plt.grid(True)
        plt.show()

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


    def gen_scores_for_sensor(self, non_norm=True, abs_val=False, force_load=False, skip_default_sensors=False, add_other=False):
        preds = Predictor.where(task_id=self.task_id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
        # print(len(preds), " existing predictors found.")
        prs = Predictor.where(task_id=self.task_id, multi_predictor_id=self.id, cohort_id=self.cohort_id)
        for el in prs:
            el.update(accuracies=None)
        
        self.gen_items_for_sensors()

        if skip_default_sensors is True:
            curr_sensors = self.sensors
        else:
            curr_sensors = self.get_sensors()

        for sensor in curr_sensors:
            predictor = Predictor.find_or_create(task_id=self.task_id, sensor_id=sensor.id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
            predictor = predictor.train_from(force_load=force_load, add_other=add_other)
        with open('items.pickle', 'wb') as handle:
            pickle.dump(self.items, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("Done training Predictors for MP:", self.id)
    
    def get_all_preds(self):
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
        for mp in predictors:
            # Extracting the classifier accuracies
            if mp.get_accuracies() != {}:
                if alt is None:
                    accuracies = mp.get_accuracies()['classifier_accuracies']
                else:
                    accuracies = mp.get_classifier_accuracies(alt=alt)
            else:
                print("Accuracies not found:", mp.attrs())
                continue
            
            # Sorting by accuracy in descending order and rounding to three decimal places
            sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}

            print(sorted_accuracies)
            
            # Append sensor name and its sorted accuracies to the results list
            results.append((mp.sensor().name, sorted_accuracies))
        results.sort(key=lambda x: max(x[1].values()), reverse=True)

        return results
    
    def cohort_name(self):
        return Cohort.get(id=self.cohort_id).name

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

        title = self.cohort_name() + "/" + title
        
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

    
    def gen_train_combo_mp(self, model="non_abs_combo", use_norm_pred=False, norm_pred=None, get_sg_count=False):
        default_predictors = self.get_predictors()
        abs_predictors = self.get_abs_predictors()

        if use_norm_pred and norm_pred is not None:
            norm_predictors = norm_pred 
        elif use_norm_pred:
            print("using norm preds")
            norm_predictors = self.get_norm_predictors() 

        norm_dict = {pred.sensor_id: pred for pred in norm_predictors} if norm_predictors else {}

        new_mp = MultiPredictor.find_or_create(
            cohort_id=self.cohort_id, task_id=self.task_id, model=("norm_" + model) if use_norm_pred else model
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
            new_pred = Predictor.find_or_create(task_id=self.task_id, sensor_id=combo_sensor_id, multi_predictor_id=new_mp.id)
            # Pass the predictors from 'preds' list and 'new_pred' separately
            if norm_predictors:
                new_mp.combo_train(preds[0], preds[1], new_pred, norm_pred=preds[2], get_sg_count=get_sg_count)
            else:
                new_mp.combo_train(preds[0], preds[1], new_pred, get_sg_count=get_sg_count)

        print("Done!")



    def combo_train(self, default_pred, abs_pred, new_pred, norm_pred=None, classifier_name=None, remove_lateral=True, get_sg_count=False):
        # non abs non norm
        df1, y1 = default_pred.get_final_bdf(untrimmed=True, force_abs_x=remove_lateral, get_sg_count=get_sg_count)

        # abs non norm
        df2, _ = abs_pred.get_final_bdf(untrimmed=True, get_sg_count=get_sg_count)

        # norm (abs)
        df3, _ = norm_pred.get_final_bdf(untrimmed=True, get_sg_count=get_sg_count) if norm_pred else (None, None)

        comp_df = MultiPredictor.create_composite_dataframe(df1, df2, df3=df3)
        combo_df = Predictor.trim_bdf_with_boruta(comp_df, y1)    

        if len(combo_df.columns) <= 2:
            combo_df = comp_df
        
        classifiers = Predictor.define_classifiers_cls(False, False, classifier_name=classifier_name, feature_count=len(combo_df.columns))
        groups = combo_df['patient']
        

        print("current combo", combo_df)
        classifier_accuracies, classifier_params, classifier_metrics = {}, {}, {}
        scores = {}
        params = {}
        acc_metrics = {}
        for classifier_name, classifier_data in classifiers.items():
            print(f"Training {classifier_name}...")
            classifier_scores, best_params, extra_metrics, scores, params, acc_metrics = Predictor._train_from_mp(combo_df, y1, groups, classifier_name, classifier_data, scores, params, acc_metrics, mps=self, use_shap=True)
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
        pickled_df = pickle.dumps(combo_df)
        new_pred.update(aggregated_stats=pickled_df)
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