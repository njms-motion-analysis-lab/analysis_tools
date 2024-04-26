import csv
import matplotlib.pyplot as plt
from models.legacy_sensor import Sensor
import seaborn as sns
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import numpy as np
import pandas as pd

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


class MatrixPlotter(LegacyBaseModel):
    @staticmethod
    def plot_matrix(predictor_model, classifier_name, sensor_name):
        cm = np.array(predictor_model.get_accuracies()['classifier_metrics'][classifier_name]['Median Confusion Matrix'])
        sns.heatmap(cm, annot=True, cmap="Blues", vmin=0, vmax=5)
        plt.title(f'Sensor: {sensor_name} using {classifier_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()

    @staticmethod
    def plot_matrices(predictor_models, classifier_name):
        avg_cm = np.mean([np.array(model.get_accuracies()['classifier_metrics'][classifier_name]['Median Confusion Matrix']) for model in predictor_models], axis=0)
        sns.heatmap(avg_cm, annot=True, cmap="Blues", vmin=0, vmax=5)
        plt.title(f'Average for {classifier_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
    
    @staticmethod
    def show(results, results_two=None, task=None, h_one=None, h_two=None, alt=False):
        return MatrixPlotter.view_and_save_results(results, results_two=results_two,task=task, h_one=h_one,h_two=h_two, alt=alt)

    @staticmethod
    def view_and_save_results(results, results_two=None, task=None, h_one=None, h_two=None, alt=False):
        # For single heatmap plotting or subplots, set vmin and vmax directly
        
        
        sensor_names = {
            'rfin': 'Hand',
            'rbhd': 'Back of Head',
            'rfrm': 'Forearm',
            'rwra': 'Wrist A',
            'rwrb': 'Wrist B',
            'relbm': 'Medial Elbow',
            'rfhd': 'Front of Head',
            'relb': 'Elbow',
            'rsho': 'Shoulder',
            'rupa': 'Upper Arm'
        }
        if alt:
            results = sorted(results, key=lambda x: SENSOR_CODES.index(x[0]))

        # Convert the lists to dictionaries
        results_dict = {sensor: accuracies for sensor, accuracies in results}
        results_two_dict = {sensor: accuracies for sensor, accuracies in results_two} if results_two else {}

        # Create an ordered list of keys from results that are also in results_two
        ordered_keys = [key for key, _ in results if key in results_two_dict]
        # Create an ordered list of keys from results that are also in results_two
        if alt:
            ordered_keys = SENSOR_CODES

        # Reorder results_two based on the order of keys in results
        reordered_results_two = sorted(results_two, key=lambda x: ordered_keys.index(x[0]))
        # Convert the reordered list to a dictionary
        results_two_dict = {sensor: accuracies for sensor, accuracies in reordered_results_two}

        # Synchronizing keys in both dictionaries
        keys_in_results_one = set(results_dict.keys())
        keys_in_results_two = set(results_two_dict.keys())

        for key in keys_in_results_one - keys_in_results_two:
            del results_dict[key]

        for key in keys_in_results_two - keys_in_results_one:
            del results_two_dict[key]

        # Get all model names
        all_models = set()
        for accuracies in results_dict.values():
            all_models.update(accuracies.keys())
        all_models = sorted(list(all_models))

        # Remove unwanted models
        unwanted_models = ['LightGBM', 'NaiveBayes', 'NeuralNetwork']
        for model in unwanted_models:
            try:
                all_models.remove(model)
            except ValueError:
                pass  # Model not in the list, so simply pass
            
            for sensor, accuracies in results_dict.items():
                row = [sensor.replace('_x', '')]
                for model in all_models:
                    row.append(accuracies.get(model, 'N/A'))  # Appending model accuracy, if not found write 'N/A'

        # Convert the results into a dataframe for easy plotting
        df1 = pd.DataFrame(columns=['Sensor'] + all_models)
        
        for sensor, accuracies in results_dict.items():
            row = {'Sensor': sensor.replace('_x', '')}
            for model in all_models:
                row[model] = accuracies.get(model, None)
            df1 = df1.append(row, ignore_index=True)
        
        # Order columns by their highest score
        ordered_columns = ['Sensor'] + df1.drop('Sensor', axis=1).mean().sort_values(ascending=False).index.tolist()
        df1 = df1[ordered_columns]
        
        if results_two is not None:
            df2 = pd.DataFrame(columns=['Sensor'] + all_models)
            for sensor, accuracies in results_two_dict.items():
                row = {'Sensor': sensor.replace('_x', '')}
                for model in all_models:
                    row[model] = accuracies.get(model, None)
                df2 = df2.append(row, ignore_index=True)
            

            ordered_columns = ['Sensor'] + df2.drop('Sensor', axis=1).mean().sort_values(ascending=False).index.tolist()
            df2 = df2[ordered_columns]
            
            # Create a subplot with 1 row and 2 columns
            # fig, ax = plt.subplots(1, 2, figsize=(24, 8))
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
            df1['Sensor'] = df1['Sensor'].map(sensor_names)
            df2['Sensor'] = df2['Sensor'].map(sensor_names)

            cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            if alt is False:
                vmin = min(df1.drop('Sensor', axis=1).astype(float).min().min(), df2.drop('Sensor', axis=1).astype(float).min().min())
                vmax = max(df1.drop('Sensor', axis=1).astype(float).max().max(), df2.drop('Sensor', axis=1).astype(float).max().max())
            else:
                vmin, vmax = 0.40, 1.00  # Set these values for the heatmap
            sns.heatmap(df1.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, yticklabels=1, ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
            if h_one != None:
                ax1.set_title(h_one)
            ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
            ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
            
            sns.heatmap(df2.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, yticklabels=1, ax=ax2, vmin=vmin, vmax=vmax, cbar_ax=cax)
            if h_two != None:
                ax2.set_title(h_two)
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
            
            # plt.tight_layout(rect=[0, 0, 0.9, 1])
            plt.subplots_adjust(top=0.94, bottom=0.15)  # Adjust the top and bottom margins

            plt.show()

        else:
            # Heatmap
            plt.figure(figsize=(12, 8))
            sns.heatmap(df1.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, cbar_kws={'label': 'Accuracy'}, yticklabels=1)
            if task is not None:
                plt.title(f"Model Accuracies Across Sensors for {task} Task")
            else:
                plt.title(f"Model Accuracies Across Sensors")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            plt.show()