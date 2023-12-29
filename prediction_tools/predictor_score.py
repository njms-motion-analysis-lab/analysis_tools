import pickle
import sqlite3
import shap
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from ts_fresh_params import get_params_for_column, PARAMS

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
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        for col_name in combined_X_np_cols:
            print(col_name)
            print(get_params_for_column(col_name))

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
            # Make sure directory exists or create it
                    # Make sure directory exists or create it
            if not abs_val:
                directory_path = "generated_pngs/shap_beeswarm/"
            else:
                directory_path = "generated_pngs/shap_beeswarm/abs/"
            
            if non_norm == False:
                directory_path = "generated_pngs/shap_beeswarm/non_norm/"
    
            if title != None:
                directory_path = (directory_path + title + "/")

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

    def get_info(self):
        from importlib import import_module
        Predictor = import_module("prediction_tools.legacy_predictor").Predictor
        pr = Predictor.get(self.predictor_id)
        task = Task.get(pr.task_id).description
        name = Sensor.get(pr.sensor_id).name
        return task + '_' + name + '_' + self.classifier_name

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