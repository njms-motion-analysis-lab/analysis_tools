import csv
import matplotlib.pyplot as plt
from models.legacy_sensor import Sensor
import seaborn as sns
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
import numpy as np
import pandas as pd

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
    def view_and_save_results(results, results_two=None, task=None):
        # Convert the list to a dictionary
        results_dict = {sensor: accuracies for sensor, accuracies in results}
        if results_two is not None:
            results_two_dict = {sensor: accuracies for sensor, accuracies in results_two}

        # Get all model names
        all_models = set()
        for accuracies in results_dict.values():
            all_models.update(accuracies.keys())
        all_models = sorted(list(all_models))

        # Remove unwanted models
        unwanted_models = ['LightGBM', 'NaiveBayes']
        for model in unwanted_models:
            try:
                all_models.remove(model)
            except ValueError:
                pass  # Model not in the list, so simply pass

        # Write to CSV
        with open('output.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Sensor'] + all_models)  # Writing header
            
            for sensor, accuracies in results_dict.items():
                row = [sensor.replace('_x', '')]
                for model in all_models:
                    row.append(accuracies.get(model, 'N/A'))  # Appending model accuracy, if not found write 'N/A'
                writer.writerow(row)

        # Convert the results into a dataframe for easy plotting
        df = pd.DataFrame(columns=['Sensor'] + all_models)

        for sensor, accuracies in results_dict.items():
            row = {'Sensor': sensor.replace('_x', '')}
            for model in all_models:
                row[model] = accuracies.get(model, None)
            df = df.append(row, ignore_index=True)

        # Order columns by their highest score
        ordered_columns = ['Sensor'] + df.drop('Sensor', axis=1).max().sort_values(ascending=False).index.tolist()
        if results_two is not None:
            for sensor, accuracies in results_two_dict.items():
                row = {'Sensor': sensor.replace('_x', '')}
                for model in all_models:
                    row[model] = accuracies.get(model, None)
                df = df.append(row, ignore_index=True)
        

        df = df[ordered_columns]

        # Heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, cbar_kws={'label': 'Accuracy'}, yticklabels=1)
        if task is not None and results_two is None:
            plt.title(f"Model Accuracies Across Sensors for {task} Task")  # Using provided method to get title
        else:
            plt.title(f"Model Accuracies Across Sensors")  # Using provided method to get title

        plt.xticks(rotation=45)
        plt.yticks(rotation=0)  # This ensures that sensor names are horizontal
        plt.show()