import pickle
import sqlite3
import shap
import pandas as pd
import os
import matplotlib.pyplot as plt
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel

class PredictorScore(LegacyBaseModel):
    table_name = "predictor_score"

    def __init__(self, id=None, classifier_name=None, score_type=None, matrix=None, classifier_id=None, predictor_id=None, created_at=None, updated_at=None):
        self.id = id
        self.classifier_name = classifier_name
        self.score_type = score_type
        self.matrix = matrix
        self.classifier_id = classifier_id
        self.predictor_id = predictor_id
        self.created_at = created_at
        self.updated_at = updated_at


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

    # def view_shap_plot(self):
    #     # Fetch the serialized data
    #     aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

    #     # Convert numpy array back to DataFrame
    #     combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)  # We're using the global `combined_X` variable for columns

    #     # Call shap.summary_plot
    #     shap.summary_plot(aggregated_shap_values_retrieved, features=combined_X_df, feature_names=combined_X_np_cols)




    def view_shap_plot(self, title=None):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Convert the retrieved SHAP values and feature values into an Explanation object
        explanation = shap.Explanation(values=aggregated_shap_values_retrieved, data=combined_X_df)
        # Make sure directory exists or create it
        directory_path = "generated_pngs/shap_beeswarm/"
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Construct the filename from the provided title
        filename = os.path.join(directory_path, title.replace(" ", "_") + ".png")

        # Create a new figure explicitly
        plt.figure()

        # Call shap.summary_plot with the Explanation object
        shap.plots.beeswarm(explanation, show=False)
        
        # Save the current active figure as PNG
        plt.savefig(filename, bbox_inches='tight')
        
        # Optionally, close the plot to free up resources
        plt.close()

    def view_shap_heatmap(self, title=None):
        # Fetch the serialized data
        aggregated_shap_values_retrieved, combined_X_np, combined_X_np_cols = self.get_matrix("matrix")

        # Convert numpy array back to DataFrame
        combined_X_df = pd.DataFrame(combined_X_np, columns=combined_X_np_cols)

        # Convert the retrieved SHAP values and feature values into an Explanation object
        explanation = shap.Explanation(values=aggregated_shap_values_retrieved, data=combined_X_df, 
                               feature_names=combined_X_np_cols)
        # Make sure directory exists or create it
        directory_path = "generated_pngs/shap_heatmap/"
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
            

