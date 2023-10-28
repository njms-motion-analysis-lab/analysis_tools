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
        sensor_names = {
            'rfin': 'Finger',
            'rbhd': 'Back of Hand',
            'rfrm': 'Forearm',
            'rwra': 'Wrist A',
            'rwrb': 'Wrist B',
            'relbm': 'Med. Elbow',
            'rfhd': 'Front of Hand',
            'relb': 'Elbow',
            'rsho': 'Shoulder',
            'rupa': 'Upper Arm'
        }
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
        unwanted_models = ['LightGBM', 'NaiveBayes', 'NeuralNetwork']
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
                vmin = min(df1.drop('Sensor', axis=1).astype(float).min().min(), df2.drop('Sensor', axis=1).astype(float).min().min())
                vmax = max(df1.drop('Sensor', axis=1).astype(float).max().max(), df2.drop('Sensor', axis=1).astype(float).max().max())

                sns.heatmap(df1.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, yticklabels=1, ax=ax1, vmin=vmin, vmax=vmax, cbar=False)
                ax1.set_title(f"Healthy")
                ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)
                ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0)
                
                sns.heatmap(df2.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, yticklabels=1, ax=ax2, vmin=vmin, vmax=vmax, cbar_ax=cax)
                ax2.set_title(f"Cerebral Palsy")
                ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
                ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)
                
                # plt.tight_layout(rect=[0, 0, 0.9, 1])


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