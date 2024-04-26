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
        # Define sensor names mapping
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

        # Helper function to process and prepare DataFrame from results
        def prepare_df(results_dict, sensor_names, all_models):
            # Create DataFrame from results
            df = pd.DataFrame.from_dict(results_dict, orient='index', columns=all_models)
            df.index = df.index.map(sensor_names)  # Map sensor codes to names
            df['Average'] = df.mean(axis=1)  # Calculate row averages
            # Append column averages as a new row
            col_avg = df.mean().to_dict()
            col_avg['Average'] = df['Average'].mean()  # Calculate the overall average
            df = df.append(pd.DataFrame(col_avg, index=['Column Average']))
            return df

        # Convert the lists to dictionaries and filter models
        results_dict = {sensor: accuracies for sensor, accuracies in results}
        results_two_dict = {sensor: accuracies for sensor, accuracies in results_two} if results_two else {}

        # Get all model names excluding unwanted models
        unwanted_models = ['LightGBM', 'NaiveBayes', 'NeuralNetwork']
        all_models = set(model for accuracies in results_dict.values() for model in accuracies.keys() if model not in unwanted_models)
        all_models = sorted(list(all_models))

        # Prepare DataFrames
        df1 = prepare_df(results_dict, sensor_names, all_models)
        df2 = prepare_df(results_two_dict, sensor_names, all_models) if results_two else None

        # Plotting
        if df2 is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.2})
            vmin = min(df1.min().min(), df2.min().min())
            vmax = max(df1.max().max(), df2.max().max())
        else:
            fig, ax1 = plt.subplots(figsize=(12, 8))
            vmin, vmax = df1.min().min(), df1.max().max()

        sns.heatmap(df1.astype(float), cmap="YlGnBu", annot=True, vmin=vmin, vmax=vmax, ax=ax1, cbar=df2 is None)
        ax1.set_title(h_one if h_one else "Heatmap 1")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        if df2 is not None:
            sns.heatmap(df2.astype(float), cmap="YlGnBu", annot=True, vmin=vmin, vmax=vmax, ax=ax2, cbar=True)
            ax2.set_title(h_two if h_two else "Heatmap 2")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        plt.tight_layout()
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