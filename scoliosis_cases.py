

from prediction_tools.scoliosis_time_predictor import ScoliosisTimePredictor
from statsmodels.stats.proportion import proportions_ztest, confint_proportions_2indep
from sklearn.pipeline import Pipeline
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import os
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from xgboost import XGBClassifier
from statsmodels.stats.proportion import proportions_ztest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import numpy as np
import pandas as pd

# Configure classification models with proper balancing
CLASSIFICATION_MODELS = {
    "RandomForestClassifier": {
        "classifier": RandomForestClassifier(class_weight='balanced'),
        "param_grid": {
            "classifier__n_estimators": [50, 150],
            "classifier__max_depth": [1, 3, 5, 7],
            'classifier__min_samples_split': [2, 3],
            'classifier__min_samples_leaf': [1, 2, 3,],
            'classifier__max_features': ['sqrt', 'log2']
        }
    },
    # "LogisticRegression": {
    #     "classifier": LogisticRegression(class_weight='balanced', max_iter=1000),
    #     "param_grid": {
    #         "classifier__C": [0.1, 1, 10],
    #         "classifier__penalty": ["l1", "l2"],
    #         "classifier__solver": ["liblinear"]
    #     }
    # },
    # # In model definitions:
    # "RandomForestClassifier": {
    #     "classifier": RandomForestClassifier(class_weight='balanced'),
    #     "param_grid": {
    #         "classifier__max_depth": [3, 5],  # Limit complexity
    #         "classifier__min_samples_leaf": [10]  # Prevent overfitting
    #     }
    # },
    # "XGBClassifier": {
    #     "classifier": XGBClassifier(random_state=42, eval_metric='logloss'),
    #     "param_grid": {
    #         "classifier__n_estimators": [100, 200],
    #         "classifier__max_depth": [3, 6],
    #         "classifier__learning_rate": [0.01, 0.1],
    #         "classifier__scale_pos_weight": [1, 5, 10]  # Auto-calculate this later
    #     }
    # }
}

RAW_DATA_FOLDER = "raw_data/scoliosis_cases"

for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
    for filename in filenames:
        # Initialize predictor with proper CSV path
        csv_path = os.path.join(subdir, filename)
        import pdb;pdb.set_trace()
        stp = ScoliosisTimePredictor(csv_path=csv_path)
        
        # Generate and filter dataframe
        df = stp.generate_training_dataframe(target_col="doptodis")

        # =============================================================================
        # UPDATED: Instead of keeping columns that start with "abx_regimen", we now
        # keep those that begin with either "combo_abx_regimen_" or "single_abx_regimen_"
        # =============================================================================
        keep_cols = ['any_ssi'] + [c for c in df.columns 
                                   if c.startswith('combo_abx_regimen_') or c.startswith('single_abx_regimen_')]
        df = df[keep_cols]  # or merge with other desired columns

        # Calculate class weights dynamically
        positive_count = df['any_ssi'].sum()
        negative_count = len(df) - positive_count
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1
        
        # (If using XGBoost, update its parameters with the scale_pos_weight)
        # CLASSIFICATION_MODELS['XGBClassifier']['param_grid']['classifier__scale_pos_weight'] = [scale_pos_weight]

        # Run classification pipeline
        best_pipeline, best_metrics, best_model_name = stp.grid_search_pipeline(
            data=df,
            target_column="any_ssi",
            models=CLASSIFICATION_MODELS  # Pass your classification models
        )

        best_pipeline, best_metrics, best_model_name = stp.random_forest_pipeline_with_shap(data=df)


        # Prepare data with stratification
        X_train, X_test, y_train, y_test = stp.prepare_train_test(df, "any_ssi")

        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X_train, y_train)

        # SHAP explanation
        model = best_pipeline.named_steps['classifier']
        preprocessor = Pipeline(best_pipeline.steps[:-1]) if len(best_pipeline.steps) > 1 else None
        
        if preprocessor:
            X_test_transformed = preprocessor.transform(X_test)
        else:
            X_test_transformed = X_test
        
        if isinstance(X_test_transformed, np.ndarray):
            X_test_transformed = pd.DataFrame(
                X_test_transformed,
                columns=best_pipeline[:-1].get_feature_names_out()
            )

            # --- After computing shap_values, e.g.:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)

        # --- After computing shap_values and filtering regimen columns:
        # Get all regimen columns from the model features:
        abx_cols = [col for col in X_test_transformed.columns 
                    if col.startswith('single_abx_regimen_') or col.startswith('combo_abx_regimen_')]
        abx_cols = [col for col in abx_cols if '.1' not in col]

        # Split into single-agent and combination columns:
        single_cols = [col for col in abx_cols if col.startswith('single_abx_regimen_')]
        combo_cols  = [col for col in abx_cols if col.startswith('combo_abx_regimen_')]

        # ----------------------------------------------------------------
        # Print out the SHAP values for each included antibiotic regimen column:
        print("SHAP values for each antibiotic regimen column:")
        for col in abx_cols:
            try:
                idx = X_test_transformed.columns.get_loc(col)
                mean_val = np.mean(shap_values[:, idx])
                std_val = np.std(shap_values[:, idx])
                print(f"{col}: mean = {mean_val:.4f}, std = {std_val:.4f}")
            except Exception as e:
                print(f"Error computing SHAP for {col}: {e}")
        # ----------------------------------------------------------------

        # ---------------------------
        # NEW: Display SHAP beeswarm plot for combination regimen columns
        # ---------------------------
        if combo_cols:
            print("Displaying SHAP beeswarm plot for combination regimens...")
            # Get indices for combo columns
            combo_feature_indices = [X_test_transformed.columns.get_loc(c) for c in combo_cols]
            # Subset the SHAP values and feature matrix for combination features only
            combo_shap_values = shap_values[:, combo_feature_indices]
            combo_features = X_test_transformed[combo_cols]
            # Produce a beeswarm plot (plot_type "dot" is the default for beeswarm)
            shap.summary_plot(combo_shap_values, combo_features, plot_type="dot", show=True)
        # ---------------------------

        antibiotic_report = []
        # Split into single-agent and combination columns:
        single_cols = [col for col in abx_cols if col.startswith('single_abx_regimen_')]
        combo_cols  = [col for col in abx_cols if col.startswith('combo_abx_regimen_')]

        # Process Single-Agent Regimens:
        if len(single_cols) > 0:
            # Use 'single_abx_regimen_cef_alone' if present; otherwise choose the one with the highest count.
            reference_regimen_single = 'single_abx_regimen_cef_alone'
            if reference_regimen_single not in single_cols:
                ref_counts = {col: df[col].sum() for col in single_cols}
                reference_regimen_single = max(ref_counts, key=ref_counts.get)
                print(f"Reference single regimen not found; using {reference_regimen_single} as reference for single-agent report.")
            ref_mask = df[reference_regimen_single] == 1
            ref_ssi = df.loc[ref_mask, 'any_ssi'].sum()
            ref_total = ref_mask.sum()
            
            for abx in single_cols:
                if abx == reference_regimen_single:
                    continue
                try:
                    idx = X_test_transformed.columns.get_loc(abx)
                    shap_mean = np.mean(shap_values[:, idx])
                    shap_std = np.std(shap_values[:, idx])
                except Exception:
                    shap_mean = np.nan
                    shap_std = np.nan
                
                abx_mask = df[abx] == 1
                abx_ssi = df.loc[abx_mask, 'any_ssi'].sum()
                abx_total = abx_mask.sum()
                
                if abx_total < 4 or ref_total < 4:
                    continue
                
                or_ = (abx_ssi / abx_total) / (ref_ssi / ref_total)
                ci_lower, ci_upper = confint_proportions_2indep(abx_ssi, abx_total, ref_ssi, ref_total)
                stat, pval = proportions_ztest([abx_ssi, ref_ssi], [abx_total, ref_total])
                p1 = abx_ssi / abx_total
                p2 = ref_ssi / ref_total
                effect_size = proportion_effectsize(p1, p2)
                power_analysis = NormalIndPower()
                try:
                    power = power_analysis.solve_power(
                        effect_size=effect_size,
                        nobs1=abx_total,
                        alpha=0.05,
                        ratio=(ref_total / abx_total),
                        alternative='two-sided'
                    )
                except:
                    power = np.nan
                
                antibiotic_report.append({
                    'Regimen_Type': 'Single',
                    'Antibiotic': abx.replace("single_abx_regimen_", "").upper(),
                    'SHAP_Mean': shap_mean,
                    'SHAP_Std': shap_std,
                    'ABX_SSI': abx_ssi,
                    'ABX_Total': abx_total,
                    'Odds_Ratio': or_,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'P_Value': pval,
                    'Power': power,
                    'SSI_Rate_Regimen': f"{abx_ssi / abx_total:.1%}",
                    'SSI_Rate_Reference': f"{ref_ssi / ref_total:.1%}"
                })
        else:
            print("No single-agent regimen columns found; skipping single-agent report.")

        # Process Combination Regimens:
        if len(combo_cols) > 0:
            reference_regimen_combo = 'combo_abx_regimen_cefazolin'
            if reference_regimen_combo not in combo_cols:
                ref_counts = {col: df[col].sum() for col in combo_cols}
                reference_regimen_combo = max(ref_counts, key=ref_counts.get)
                print(f"Reference combination regimen not found; using {reference_regimen_combo} as reference for combo report.")
            ref_mask = df[reference_regimen_combo] == 1
            ref_ssi = df.loc[ref_mask, 'any_ssi'].sum()
            ref_total = ref_mask.sum()
            
            for abx in combo_cols:
                if abx == reference_regimen_combo:
                    continue
                try:
                    idx = X_test_transformed.columns.get_loc(abx)
                    shap_mean = np.mean(shap_values[:, idx])
                    shap_std = np.std(shap_values[:, idx])
                except Exception:
                    shap_mean = np.nan
                    shap_std = np.nan
                
                abx_mask = df[abx] == 1
                abx_ssi = df.loc[abx_mask, 'any_ssi'].sum()
                abx_total = abx_mask.sum()
                
                if abx_total < 4 or ref_total < 4:
                    continue
                
                or_ = (abx_ssi / abx_total) / (ref_ssi / ref_total)
                ci_lower, ci_upper = confint_proportions_2indep(abx_ssi, abx_total, ref_ssi, ref_total)
                stat, pval = proportions_ztest([abx_ssi, ref_ssi], [abx_total, ref_total])
                p1 = abx_ssi / abx_total
                p2 = ref_ssi / ref_total
                effect_size = proportion_effectsize(p1, p2)
                power_analysis = NormalIndPower()
                try:
                    power = power_analysis.solve_power(
                        effect_size=effect_size,
                        nobs1=abx_total,
                        alpha=0.05,
                        ratio=(ref_total / abx_total),
                        alternative='two-sided'
                    )
                except:
                    power = np.nan
                
                antibiotic_report.append({
                    'Regimen_Type': 'Combo',
                    'Antibiotic': abx.replace("combo_abx_regimen_", "").upper(),
                    'SHAP_Mean': shap_mean,
                    'SHAP_Std': shap_std,
                    'ABX_SSI': abx_ssi,
                    'ABX_Total': abx_total,
                    'Odds_Ratio': or_,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper,
                    'P_Value': pval,
                    'Power': power,
                    'SSI_Rate_Regimen': f"{abx_ssi / abx_total:.1%}",
                    'SSI_Rate_Reference': f"{ref_ssi / ref_total:.1%}"
                })
        else:
            print("No combination regimen columns found; skipping combination report.")

        # Combine and sort the antibiotic report
        if antibiotic_report:
            report_df = pd.DataFrame(antibiotic_report)
            report_df = report_df.sort_values('SHAP_Mean', key=lambda x: np.abs(x), ascending=False)
            
            
            print("\nClinical Antibiotic Regimen Report:")
            print(report_df.to_markdown(index=False, floatfmt=("", "", ".3f", ".3f", ".0f", ".0f", ".3f", ".3f", ".3f", ".3f", ".1%", ".1%", ".3f", ".1%")))
            # print(report_df.to_markdown(index=False, floatfmt=".3f"))
            
            # Apply Bonferroni correction to p-values:
            report_df['Adj_P_Value'] = report_df['P_Value'] * len(report_df)
            report_df.to_csv(f"antibiotic_report_{filename.split('.')[0]}.csv", index=False)
        else:
            print(f"\nInsufficient data for clinical comparison in {filename}.")




