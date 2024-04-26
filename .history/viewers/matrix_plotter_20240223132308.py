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
            'rfin_x': 'Hand',
            'rbhd_x': 'Back of Head',
            'rfrm_x': 'Forearm',
            'rwra_x': 'Wrist A',
            'rwrb_x': 'Wrist B',
            'relbm_x': 'Medial Elbow',
            'rfhd_x': 'Front of Head',
            'relb_x': 'Elbow',
            'rsho_x': 'Shoulder',
            'rupa_x': 'Upper Arm'
        }

        # Helper function to prepare DataFrame from results
        def prepare_df(results_dict, all_models):
            df = pd.DataFrame.from_dict(results_dict, orient='index', columns=all_models)
            # Calculate row averages and append to sensor names
            row_avgs = df.mean(axis=1)
            df.index = [f"{sensor_names.get(idx, idx)} ({avg:.3f})" for idx, avg in row_avgs.iteritems()]
            # Calculate column averages and append to model names
            col_avgs = df.mean()
            df.columns = [f"{col} ({avg:.3f})" for col, avg in col_avgs.iteritems()]
            return df

        # Convert the lists to dictionaries and filter models
        results_dict = {sensor: accuracies for sensor, accuracies in results}
        results_two_dict = {sensor: accuracies for sensor, accuracies in results_two} if results_two else {}

        # Get all model names excluding unwanted models
        unwanted_models = ['LightGBM', 'NaiveBayes', 'NeuralNetwork']
        all_models = set(model for accuracies in results_dict.values() for model in accuracies.keys() if model not in unwanted_models)
        all_models = sorted(list(all_models))

        # Prepare DataFrames
        df1 = prepare_df(results_dict, all_models)
        df2 = prepare_df(results_two_dict, all_models) if results_two else None

        # Plotting
        if df2 is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8), gridspec_kw={'width_ratios': [1, 1]})
            vmin = min(df1.min().min(), df2.min().min())
            vmax = max(df1.max().max(), df2.max().max())
        else:
            fig, ax1 = plt.subplots(figsize=(12, 8))
            vmin, vmax = df1.min().min(), df1.max().max()

        sns.heatmap(df1.astype(float), cmap="YlGnBu", annot=True, vmin=vmin, vmax=vmax, ax=ax1, cbar=df2 is None)
        ax1.set_title(h_one if h_one else "Heatmap 1")
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)

        if df2 is not None:
            sns.heatmap(df2.astype(float), cmap="YlGnBu", annot=True, vmin=vmin, vmax=vmax, ax=ax2, cbar=True)
            ax2.set_title(h_two if h_two else "Heatmap 2")
            ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
            ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

        plt.tight_layout()
        plt.show()


            #        if alt is False:
            #     vmin = min(df1.drop('Sensor', axis=1).astype(float).min().min(), df2.drop('Sensor', axis=1).astype(float).min().min())
            #     vmax = max(df1.drop('Sensor', axis=1).astype(float).max().max(), df2.drop('Sensor', axis=1).astype(float).max().max())
            # else:
            #     vmin, vmax = 0.40, 1.00  # Set these values for the heatmap