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
        
        # Assuming SENSOR_CODES is defined elsewhere in your class
        if alt:
            results = sorted(results, key=lambda x: SENSOR_CODES.index(x[0]))

        # Helper function to process and prepare DataFrame from results
        def prepare_df(results_dict, all_models, is_alt):
            # Create DataFrame from results
            df = pd.DataFrame.from_dict(results_dict, orient='index', columns=all_models)
            if is_alt:  # If alt is True, use SENSOR_CODES to order the rows
                df = df.reindex(SENSOR_CODES)
            df = df.rename(index=sensor_names)  # Map sensor codes to names
            df['Average'] = df.mean(axis=1)  # Calculate row averages
            df.loc['Column Average', :] = df.mean()  # Calculate column averages
            # Append ' (Average)' to the row and column names
            df.index = [f'{i} ({avg:.3f})' if i != 'Column Average' else i for i, avg in zip(df.index, df['Average'])]
            df.columns = [f'{i} ({avg:.3f})' if i != 'Average' else i for i, avg in zip(df.columns, df.mean())]
            return df

        # Convert the lists to dictionaries and filter models
        results_dict = {sensor: accuracies for sensor, accuracies in results}
        results_two_dict = {sensor: accuracies for sensor, accuracies in results_two} if results_two else {}

        # Get all model names excluding unwanted models
        unwanted_models = ['LightGBM', 'NaiveBayes', 'NeuralNetwork']
        all_models = sorted({model for accuracies in results_dict.values() for model in accuracies if model not in unwanted_models})

        # Prepare DataFrames
        df1 = prepare_df(results_dict, all_models, alt)
        df2 = prepare_df(results_two_dict, all_models, alt) if results_two else None

        # Plotting
        fig, axs = plt.subplots(1, 2, figsize=(24, 8)) if df2 is not None else plt.subplots(figsize=(12, 8))
        cbar_ax = fig.add_axes([.91, .3, .03, .4]) if df2 is not None else None

        sns.heatmap(df1.astype(float), cmap="YlGnBu", annot=True, ax=axs[0] if df2 is not None else axs, cbar_ax=cbar_ax)
        axs[0].set_title(h_one if h_one else "Heatmap 1")
        axs[0].set_xticklabels(axs[0].get_xticklabels(), rotation=45)
        axs[0].set_yticklabels(axs[0].get_yticklabels(), rotation=0)

        if df2 is not None:
            sns.heatmap(df2.astype(float), cmap="YlGnBu", annot=True, ax=axs[1], cbar_ax=cbar_ax)
            axs[1].set_title(h_two if h_two else "Heatmap 2")
            axs[1].set_xticklabels(axs[1].get_xticklabels(), rotation=45)
            axs[1].set_yticklabels(axs[1].get_yticklabels(), rotation=0)
            plt.subplots_adjust(right=0.9)  # Adjust right edge to make room for colorbar

        plt.show()