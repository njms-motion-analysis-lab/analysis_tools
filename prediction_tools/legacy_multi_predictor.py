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
from prediction_tools.legacy_predictor import Predictor
import matplotlib.colors as mcolors
from prediction_tools.predictor_score import PredictorScore
from viewers.matrix_plotter import MatrixPlotter
from viewers.multi_plotter import MultiPlotter


SENSOR_CODES = [
    'rfin_x',
    'rwra_x',
    'rwrb_x',
    'rfrm_x',
    'relb_x',
    'relbm_x',
    'rupa_x',
    'rsho_x',
    'rbhd_x',
    'rfhd_x',
]

NUM_TOP = 10
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
        

    def gen_items_for_sensor(self, snr, ntaf=True):
        if not ntaf:
            print(f"Generating normed, abs score, for task: {self.task.description}")
            nfat = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=True, cohort_id=self.cohort_id)
            nfat.train_it()
            print(f"Generating normed, non abs score, for task: {self.task.description}")
            nfaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=False, abs_val=False, cohort_id=self.cohort_id)
            nfaf.train_it()
            print(f"Generating non normed, abs score, for task: {self.task.description}")
            ntat = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=True, abs_val=True, cohort_id=self.cohort_id)
            # ntat.train_it()
            print(f"Generating non normed, non abs score, for task: {self.task.description}", cohort_id=self.cohort_id)
        
        ntaf = Predictor.find_or_create(task_id=self.task_id, sensor_id=snr.id, non_norm=True, abs_val=True, cohort_id=self.cohort_id)
        ntaf.train_it()
        return ntaf
    

    def get_sensors(self):
        return Sensor.where(name=SENSOR_CODES)


    def get_predictors(self, abs_val=False, non_norm=True):
        return Predictor.where(multi_predictor_id=self.id, abs_val=abs_val, non_norm=non_norm)


    def get_norm_predictors(self):
        return Predictor.where(multi_predictor_id=self.id, non_norm=0)


    def get_abs_predictors(self):
        return Predictor.where(multi_predictor_id=self.id, abs_val=True, non_norm=1)


    def get_predictor_model_accuracies(self, model_name, abs_val=False, non_norm=True):
        preds = self.get_predictors(abs_val=abs_val, non_norm=non_norm)
        print("preds -001", preds)
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

        return preds
    
    def show_predictor_scores(self, models, abs_val=False, non_norm=True, reverse_order=False, first_model_features=None):
        pred_kit = []
        for m in models:
            pred_kit.append(
                self.get_predictor_scores_for_model(
                    m, sort_by_sensor=True, reverse_sensor_order=reverse_order, abs_val=abs_val, non_norm=non_norm,
                )
            )
        print("pred_kit", pred_kit)
        print()
        preds = pred_kit[0]
        print("first", preds)
        print()
        ps = preds[0][0]
        if first_model_features is not None:
            first_model_features = ps[0].get_top_n_features(NUM_TOP)
        
        shap_scores, top_scores, sensors = self.gen_plot(preds, first_model_features=first_model_features)
        return self.plot_shap_changes(shap_scores, top_scores, sensors, percentage_of_average=True, first_model_features=first_model_features)


    def gen_plot(self, preds, first_model_features=None):
        shap_scores = []
        top_scores = []
        sensors = []
        ps = preds[0][0]
        if first_model_features is None:
            
            tf = ps[0].get_top_n_features(NUM_TOP)
        else:
            tf = first_model_features
    
        print("third")
        for ps, acc_score, snr in preds:
            if ps == []:
                continue
            ts = ps[0].get_top_n_features(NUM_TOP)
            tsv = ps[0].get_shap_values_for_features(ts)
            sv = ps[0].get_shap_values_for_features(tf)
            print("fourth")
            shap_scores.append(sv)
            top_scores.append(tsv)
            sensors.append(snr.human_name())
        return [shap_scores, top_scores, sensors]


    def plot_shap_changes(self, shap_scores, top_scores, sensors, percentage_of_average=False, first_model_features=None):
        # If first_model_features is provided, create a color map based on its order
        if first_model_features is not None:
            color_map = plt.cm.get_cmap('viridis', len(first_model_features))  # You can choose any colormap
            color_dict = {feature: color_map(i) for i, feature in enumerate(first_model_features)}

        # Features from the first location in top_scores
        if first_model_features is not None:
            first_location_features = list(first_model_features)
        else:
            first_location_features = list(top_scores[0].keys())

        # Plotting SHAP values for each feature across sensors
        for feature in first_location_features:
            shap_values = [shap_scores[i].get(feature, 0) for i in range(len(sensors))]
            if percentage_of_average:
                # Calculate the average of top scores for each sensor
                averages = [sum(sorted(top_scores[i].values(), reverse=True)[:NUM_TOP]) / NUM_TOP for i in range(len(sensors))]
                # Normalize shap_values by the average
                shap_values = [100 * val / avg if avg != 0 else 0 for val, avg in zip(shap_values, averages)]

            # Use the color from the color map if first_model_features is provided
            line_color = color_dict.get(feature) if first_model_features else None
            feature = MultiPlotter.convert_feature_name(feature)
            plt.plot(sensors, shap_values, label=feature, color=line_color)

        plt.title("SHAP Values for Top Features Across Sensors" + (" (Percentage of Average)" if percentage_of_average else ""))
        plt.xlabel("Sensor Name")
        plt.ylabel("SHAP Value" + (" (% of Avg)" if percentage_of_average else ""))
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

    def get_shap_values_from_scores(self, model_name):
        ps = self.get_predictor_scores_for_model(model_name)

    def get_shap_values_for_model(self, model_name):
        self.get_predictor_scores_for_model
    
    def load_items(self):
        with open('items.pickle', 'rb') as handle:
            self.items = pickle.load(handle)
    
    def gen_scores_for_sensor(self, non_norm=True, abs_val=False, force_load=False):
        preds = Predictor.where(task_id=self.task_id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
        print(len(preds), " existing predictors found.")

        for sensor in self.get_sensors():
            predictor = Predictor.find_or_create(task_id=self.task_id, sensor_id=sensor.id, non_norm=non_norm, abs_val=abs_val, multi_predictor_id=self.id, cohort_id=self.cohort_id)
            predictor = predictor.train_from(force_load=True)
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
    
    def get_all_acc(self, alt=False):
        pr = self.get_all_preds()
        return self.get_acc(preds=pr, alt=alt)
    
    def get_acc(self, non_norm=True, abs_val=False, all=False, preds=None, alt=None):
        if all:
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

    def save_shap_values(self, abs_val=False, non_norm=True, title=None):
        pr = self.get_predictors(abs_val=abs_val, non_norm=non_norm)
        
        for pred in pr:
            scores = pred.get_predictor_scores()
            for score in scores:
                if self.cohort_name != "healthy_controls":
                    print("yoolo")
                    print(self.cohort_name)
                    score.view_shap_plot(title=self.cohort_name(), abs_val=abs_val, non_norm=non_norm)
                elif title != None:
                    print("hi")
                    score.view_shap_plot(title=title, abs_val=abs_val, non_norm=non_norm)
                else:
                    print("yo")
                    score.view_shap_plot()

    
    def gen_train_combo_mp(self, model="non_abs_combo", use_norm_pred=False, norm_pred=None):
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

        for combo_sensor_id, preds in combos.items():
            print(preds)
            print("Training for sensor:", Sensor.get(combo_sensor_id).name)
            new_pred = Predictor.find_or_create(task_id=self.task_id, sensor_id=combo_sensor_id, multi_predictor_id=new_mp.id)
            # Pass the predictors from 'preds' list and 'new_pred' separately
            if norm_predictors:
                new_mp.combo_train(preds[0], preds[1], new_pred, norm_pred=preds[2])
            else:
                new_mp.combo_train(preds[0], preds[1], new_pred)

        print("Done training for sensor:", Sensor.get(combo_sensor_id).name)
        print("Done!")


    def combo_train(self, default_pred, abs_pred, new_pred, norm_pred=None, classifier_name=None, remove_lateral=True):
        # non abs non norm
        df1, y1 = default_pred.get_final_bdf(force_abs_x=remove_lateral)

        # abs non norm
        df2, _ = abs_pred.get_final_bdf()

        # norm (abs)
        df3, _ = norm_pred.get_final_bdf() if norm_pred else (None, None)

        combo_df = MultiPredictor.create_composite_dataframe(df1, df2, df3=df3)

        to_drop = Predictor.get_drop_cols(combo_df, .85)
        combo_df.drop(to_drop, axis=1, inplace=True)
        
        classifiers = Predictor.define_classifiers_cls(False, False, classifier_name=classifier_name)
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


    