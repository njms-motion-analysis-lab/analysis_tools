

from prediction_tools.scoliosis_time_predictor import ScoliosisTimePredictor
from prediction_tools.time_predictor import REGRESSION_MODELS
# RAW_DATA_FOLDER = "/raw_data/v3filteredaiscases.csv"
RAW_DATA_FOLDER = "raw_data/scoliosis_cases"
from sklearn.pipeline import Pipeline
print('hey')





import os


for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    for filename in filenames:
        stp = ScoliosisTimePredictor(csv_path="/Users/stephenmacneille/Documents/labs/raw_data/scoliosis_cases/v3filteredaiscases.csv")
        df = stp.generate_training_dataframe()
        df.to_csv("scoliosis_df_to_csv.csv", index=False)
        best_pipeline, best_metrics, best_model_name = stp.grid_search_pipeline(
            data=df,
            target_column="tothlos",
            regression_models=REGRESSION_MODELS
        )

        final_regressor = best_pipeline.named_steps["regressor"]

        # Prepare data
        X_train, X_test, y_train, y_test = stp.prepare_train_test(df, "tothlos")

        # Conditionally transform X_test if pipeline has transformations
    
        if len(best_pipeline.steps) > 1:
            # All steps except the final regressor
            transform_pipeline = Pipeline(best_pipeline.steps[:-1])
            X_test_transformed = transform_pipeline.transform(X_test)
        else:
            # Just the regressor; no transforms
            X_test_transformed = X_test

        # Use SHAP if it’s a tree-based model
        import shap
        explainer = shap.TreeExplainer(final_regressor)
        shap_values = explainer.shap_values(X_test_transformed)
        shap.summary_plot(shap_values, X_test_transformed, plot_type="dot")
        
        rf_model, metrics = stp.random_forest_pipeline_with_shap(df, target_column="tothlos")
        print("Evaluation Metrics:", metrics)
        import pdb;pdb.set_trace()
        file_path = os.path.join(subdir, filename)
        print(file_path)
        

print("Done!")





