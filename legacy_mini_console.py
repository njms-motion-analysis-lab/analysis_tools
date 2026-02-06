import datetime
from math import nan
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from migrations.legacy_table import Table as LegacyTable
import csv
import shap
import re
import pandas as pd
from models.legacy_task import Task
import os
from models.legacy_patient_task import PatientTask
from models.legacy_patient import Patient
from models.legacy_gradient_set import GradientSet
from models.legacy_position_set import PositionSet
import numpy as np
from models.legacy_sub_gradient import SubGradient
from collections import defaultdict
from models.legacy_sensor import Sensor
from sklearn.model_selection import cross_val_score, GridSearchCV
from models.legacy_cohort import Cohort
from sklearn.ensemble import RandomForestClassifier
from prediction_tools.multi_time_predictor import MultiTimePredictor
from viewers.shape_rotator import ShapeRotator
from viewers.matrix_plotter import MatrixPlotter
from prediction_tools.predictor_score import PredictorScore
import seaborn as sns
import matplotlib.pyplot as plt
from prediction_tools.legacy_predictor import Predictor
from models.legacy_task import Task
from models.legacy_sensor import Sensor
from migrations.legacy_table import Table
from models.legacy_task import Task
from datetime import datetime, timedelta
from models.legacy_trial import Trial
from prediction_tools.legacy_multi_predictor import MultiPredictor
from prediction_tools.result_compare import SigCheck
from scipy.stats import pearsonr
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor


allowed = [
    "lwra_x",
    "lwrb_x",
    "lwra_y",
    "lwrb_y",
    "lwra_z",
    "lwrb_z",
    "rwra_x",
    "rwrb_x",
    "rwra_y",
    "rwrb_y",
    "rwra_z",
    "rwrb_z",
    "rfrm_x",
    "rfrm_y",
    "rfrm_z",
    "lelb_x",
    "lelb_y",
    "lelb_z",
    "relb_x",
    "relb_y",
    "relb_z",
    "lupa_x",
    "lupa_y",
    "lupa_z",
    "rupa_x",
    "rupa_y",
    "rupa_z",
    "lfrm_x",
    "lfrm_y",
    "lfrm_z",
    'lfhd_x',
    'lfhd_y',
    'lfhd_z',
    'rfhd_x', 
    'rfhd_y', 
    'rfhd_z', 
    'rfin_x', 
    'rfin_y', 
    'rfin_z',
    'lfin_x', 
    'lfin_y', 
    'lfin_z',
]

TOP_MODELS = ['RandomForest', 'ExtraTrees', 'DecisionTree', 'CatBoost', 'XGBoost']  # Your TOP_MODELS list
classifier_names = ['RandomForest', 'ExtraTrees', 'DecisionTree', 'CatBoost', 'XGBoost']  # Your TOP_MODELS list

Table.create_tables()

import pdb;pdb.set_trace()


def replace_axis_labels(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            filepath = os.path.join(directory, filename)
            
            # Read the current title
            with open(filepath, "rb") as file:
                content = file.read()
                title_start = content.find(b"tEXtTitle")
                title_end = content.find(b"\x00", title_start)
                title = content[title_start+9:title_end].decode("latin-1")
            
            # Replace the axis portion of the title
            new_title = title.replace("x", "x (anterior-posterior)").replace("y", "y (medial-lateral)").replace("z", "z (superior-inferior)")
            
            # Modify the title in the PNG file
            modified_content = content[:title_start+9] + new_title.encode("latin-1") + content[title_end:]
            
            # Save the modified PNG file
            with open(filepath, "wb") as file:
                file.write(modified_content)


def combine_shap_beeswarm(predictor_score1, predictor_score2, title=None):
    # Retrieve SHAP values and feature data from the first PredictorScore
    shap_values1, combined_X1_np, combined_X1_cols = predictor_score1.get_matrix("matrix")
    feature_data1 = pd.DataFrame(combined_X1_np, columns=combined_X1_cols)
    
    # Retrieve SHAP values and feature data from the second PredictorScore
    shap_values2, combined_X2_np, combined_X2_cols = predictor_score2.get_matrix("matrix")
    feature_data2 = pd.DataFrame(combined_X2_np, columns=combined_X2_cols)
    
    # Intersect the feature names to find common features
    common_features = feature_data1.columns.intersection(feature_data2.columns)
    
    # Filter the SHAP values and feature data to only include the common features
    filtered_shap_values1 = shap_values1[:, feature_data1.columns.isin(common_features)]
    filtered_shap_values2 = shap_values2[:, feature_data2.columns.isin(common_features)]
    filtered_combined_X1 = feature_data1[common_features].values
    filtered_combined_X2 = feature_data2[common_features].values
    
    # Concatenate the SHAP values and feature values vertically
    combined_shap_values = np.vstack([filtered_shap_values1, filtered_shap_values2])
    combined_features = np.vstack([filtered_combined_X1, filtered_combined_X2])
    
    # Create an Explanation object with the concatenated values
    combined_explanation = shap.Explanation(values=combined_shap_values,
                                            data=combined_features,
                                            feature_names=common_features.tolist())

    # Verify that the feature names are the same for both tasks
    # if not all(feature_names1 == feature_names2):
    #     raise ValueError("The feature names between the tasks do not match.")

    # Concatenate the SHAP values and feature data for both tasks
    # combined_shap_values = np.vstack((shap_values1, shap_values2))
    # combined_feature_data = np.vstack((feature_data1, feature_data2))
    # combined_explanation = shap.Explanation(values=combined_shap_values, data=combined_feature_data, feature_names=feature_names1)
    
    # Create the directory if it doesn't exist
    directory_path = "generated_pngs/shap_beeswarm/"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

    # Construct the filename from the provided title
    filename = os.path.join(directory_path, (title or "combined_beeswarm").replace(" ", "_") + ".png")

    # Create the beeswarm plot
    plt.figure()
    shap.plots.beeswarm(combined_explanation, show=False)
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    
    # Return the filename of the saved plot for reference
    return filename


def view_shap_values(abs_val=False):
    # get cohort
    cohort_one = Cohort.where(id=1)[0]

    # get block dom prediction sets
    mps_pred_sets = MultiPredictor.where(cohort_id=cohort_one.id, task_id=3, model="default")
    print(len(mps_pred_sets))
    for mps in mps_pred_sets:
        preds = mps.get_predictors(abs_val=abs_val)
        print(preds[0].get_accuracies())
        for pred in preds:
            scores = pred.get_predictor_scores_by_classifier_names(TOP_MODELS)
            for score in scores:
                s_name = pred.sensor().name
                m_name = score.classifier_name
                t_name = pred.task().description
                plt_title = t_name + " " + s_name + " " + m_name
                score.view_shap_heatmap(title=plt_title, abs_val=abs_val)

            print(pred.get_accuracies()['classifier_metrics'])


def gather_predictor_scores_by_task(task_descriptions, sensor_name, classifier_names):
    """
    Gathers PredictorScores for specified tasks, sensor, and classifiers.
    
    Parameters:
    - task_descriptions: list of task descriptions to filter tasks
    - sensor_name: the name of the sensor to filter predictors
    - classifier_names: list of classifier names to filter PredictorScores
    
    Returns:
    A dictionary where keys are task descriptions and values are lists of PredictorScore objects.
    """
    # Gather PredictorScores by tasks
    predictor_scores_by_task = {task_desc: [] for task_desc in task_descriptions}
    relevant_pred_tasks = []

    # get cohort
    cohort_two = Cohort.where(id=2)[0]

    # get prediction sets
    mps_pred_sets = MultiPredictor.where(cohort_id=cohort_two.id)

    for mps in mps_pred_sets:
        preds = mps.get_predictors(abs_val=True)
        for pred in preds:
            
            if pred.sensor().name != sensor_name:
                continue  # Skip predictors not matching the sensor name
            
            scores = pred.get_predictor_scores_by_classifier_names(classifier_names)

            for score in scores:
                t_name = pred.task().description
                if t_name in task_descriptions:
                    relevant_pred_tasks.append(pred)
                    predictor_scores_by_task[t_name].append(score)

    # Remove any tasks that did not have scores collected
    mps.combo_train(relevant_pred_tasks[0], relevant_pred_tasks[1], 'RandomForest')
    predictor_scores_by_task = {k: v for k, v in predictor_scores_by_task.items() if v}
    
    
    return predictor_scores_by_task


def make_csv(mpss, task_name):
    # Create the directories if they do not exist
    os.makedirs(f'generated_csvs/feature_importances/{task_name}', exist_ok=True)
    
    for i, mp in enumerate(mpss):
        feature_importance_dict = mp.get_feature_importance()
        
        df = pd.DataFrame(feature_importance_dict)

        # Write the DataFrame to a CSV file
        df.to_csv(f'generated_csvs/feature_importances/{task_name}/feature_importance_sensor_{mp.sensor().name}.csv')


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


def fetch_new_tasks():
    # pd cohort
    Cohort.all()

    TASK_NAME_COLUMN = 'description'

    FEET_TOGETHER_TASK_NAME = "Balance01"
    TANDEM_TASK_NAME = "Balance02"

    HC_COHORT = "group_1_analysis_me"
    PD_COHORT = "group_2_analysis_me"
    ET_COHORT = "group_3_analysis_me"
    
    hc_cohort = Cohort.find_by("name", HC_COHORT)
    pd_cohort = Cohort.find_by("name", PD_COHORT)
    et_cohort = Cohort.find_by("name", ET_COHORT)

    # feet together balance task
    # tandem balance task
    feet_together_task = Task.find_by("description", FEET_TOGETHER_TASK_NAME)
    tandem_task = Task.find_by("description", TANDEM_TASK_NAME)

    sensor_code = "mAccelerometerX"
    sensors = Sensor.where(name=sensor_code)
    balance_tasks = [feet_together_task, tandem_task]

    exp_cohorts = [
        pd_cohort,
        # et_cohort,
        # hc_cohort,
    ]

    mpc = []

    def fix_mistake(mp):
        pra = Predictor.where(task_id=mp.task_id, cohort_id=hc_cohort.id)
        print("fixing", len(pra), "task a")
        for pa in pra:
            pa.update(multi_predictor_id=mp.id)
        
        print("done")


    for curr_cohort in exp_cohorts:
        for balance_task in balance_tasks:
            # print("Creating/Finding MP for:", balance_task.description, hc_cohort.name)
            mp = MultiPredictor.find_or_create(task_id = balance_task.id, cohort_id = curr_cohort.id)
            # print("ok im here")
            # mp.sensors = sensors
            # mp.gen_scores_for_sensor(skip_default_sensors=True, force_load=True, add_other=True)
            mpc.append(
                mp
            )
        
        
        for mp in mpc:
            mp.save_norm_shap()
            # mp.save_shap()
            # mp.save_abs_shap()
            
        
        print(len(mpc), "MP to gen found")
        for mp in reversed(mpc):
            desc = Task.get(mp.task_id).description
            sensors = mp.sensors
            print("Generating Scores for: ", desc, "Sensor: ", sensors, "PRED LEN", len(mp.get_all_preds()))
    
    print("hopefully done")
    print("done")


def gen_scores_for_mp(mp, force_load=False):
    # mp.view_progress(fix=False, multi=True)
    # mp.gen_scores_for_sensor()
    print("done with default")

    mp.gen_scores_for_sensor(non_norm=False, force_load=force_load)
    mp.gen_scores_for_sensor(force_load=force_load)
    mp.gen_scores_for_sensor(abs_val=True, force_load=force_load)
    

    norm_pred =  mp.get_norm_predictors()
    
    mp.gen_train_combo_mp(use_norm_pred=True, norm_pred=norm_pred)
    
    print("really done")


def set_dom_side():
    healthy = Patient.where(cohort_id=1)
    cp = Patient.where(cohort_id=2)


    for hp in healthy:
        hp.update(dominant_side='right')
    
    for cpp in cp:
        cpp.update(dominant_side='right')
        
    

    lefties = Patient.where(id=[21, 26, 27, 28, 56, 57, 60, 61, 62, 63])

    for lp in lefties:
        lp.update(dominant_side='left')

    pts = Patient.all()
    for pt in pts:
        print("ID", pt.id, "NAME", pt.name, "SIDE", pt.dominant_side, "COHORT_ID", pt.cohort_id)
    print(cpp.dominant_side)


def save_sp_shap_vals(mp):
    preds = mp.get_all_preds()

    mp.save_shap_values(preds=preds)


    # bc.save_shap_values(non_norm=False, title="combo")
    # rc.save_shap_values(non_norm=False, title="combo")
    print("DONE!")


def fix_preds():
    t1 = Task.where(description="Block_dominant")[0]
    t2 = Task.where(description="Block_nondominant")[0]
    tasks = [t1, t2]
    
    for task in tasks:
        mps = MultiPredictor.where(task_id=task.id)
        print(task.description)
        for mp in mps:
            preds = mp.get_all_preds()
            for pred in preds:
                ext = "good"
                if pred.cohort_id != mp.cohort_id:
                    ext = "fixed..."
                    pred.update(cohort_id=mp.cohort_id)
                print("IDS", mp.id, pred.id, "COHORT IDS (MD, PRED)", mp.cohort_id, pred.cohort_id, ext)
            print()
        
    print("OK THEN")

COHORT_NAME = "cp_before"
TASK_NAME = "Block_dominant"

def add_set_stats():
    t1 = Task.where(description="Block_dominant")[0]
    t2 = Task.where(description="Block_nondominant")[0]
    tasks = [t1, t2]
    
    for task in tasks:
        curr_task = task.description
        trials = task.trials()  # Call the method once and store the list
        num_trial = 0
        tot_trial = len(trials)
        for trial in trials:  # Iterate over the list directly
            print(trial.id)
            num_gs = 0
            len_gs = len(trial.get_gradient_sets())
            total = len_gs * tot_trial
            to_update = 0
            updated = 0
            time_threshold = datetime.now() - timedelta(hours=36)
            for gs in trial.get_gradient_sets():
                # print(gs.id, num_trial, tot_trial, num_gs, len_gs, total)
                
                # Check if the gradient set already has its statistics
                if gs.updated_at is not None:
                    # Adjust the format string to match the actual format
                    gs_updated_at = datetime.strptime(gs.updated_at, '%Y-%m-%d %H:%M:%S.%f')
                else:
                    gs_updated_at = None
            
                if not (gs.set_stats_non_norm and gs.set_stats_norm and gs.set_stats_abs) or gs_updated_at > time_threshold:
                    # print("UPDATE", gs.id, num_trial, tot_trial, num_gs, len_gs, total)
                    to_update += 1
                    # gs.gen_set_stats(force=False)  # Only force update if not already set
                
                if not (gs.set_stats_non_norm and gs.set_stats_norm and gs.set_stats_abs) or gs_updated_at > time_threshold:
                    print("UPDATE", gs.id, num_trial, tot_trial, num_gs, len_gs, total, "UPDATED", updated, "TO_UPDATE", to_update)
                    gs.gen_set_stats(force=True)  # Only force update if not already set
                    updated += 1
                
                num_gs += 1
            num_trial += 1


def make_final_combo(mp):
    norm_preds = mp.get_norm_preds()
    mp.gen_train_combo_mp(use_norm_pred=True, norm_pred=norm_preds)
    print("DONE")


def find_or_create_mtp(feature_mp=None, timing_mp=None):

    # create hc sm
    # hc sub motion 
    
    mpt = MultiTimePredictor.find_or_create(multi_predictor_feature_id = feature_mp.id, multi_predictor_id = timing_mp.id)

    # mpt.create_from_multi_predictors(
    #     feature_mp=feature_mp,
    #     timing_mp=timing_mp
    # )
    # mpt.combine_features_and_time(
    #     feature_mp=feature_mp,
    #     timing_mp=timing_mp
    # )
    ntp = mpt.get_norm_time_predictors()
    snr_results = []
    for tp in ntp:
        # TODO: Stephen, remove after debugging
        # if Sensor.get(tp.sensor_id).name in ['rfhd_x', 'rwra_x', 'rwrb_x']:
            
        results = tp.train_regression(use_log=False)
        snr_results.append([Sensor.get(tp.sensor_id).name, results])


def display_for_paper(mpa=None, cp_mpa=None, hc_combo=None, cp_combo=None, hcgsmp=None, cpgsmp=None, hc_set_list=None, cp_set_list=None):

    import pdb;pdb.set_trace()

    
    mpa.save_shap_values(preds=mpa.get_norm_preds(), title="hc_norm_6")
    cp_mpa.save_shap_values(preds=cp_mpa.get_norm_preds(), title="cp_norm_6")
    hcgsmp.save_shap_values(preds=hcgsmp.get_norm_preds(), title="gs_hc_norm_6")
    cpgsmp.save_shap_values(abs_val=False, non_norm=False, preds=cpgsmp.get_norm_preds(), title="gs_cp_norm_6")
    print("ok shaps")
    if mpa and cp_mpa:
        MatrixPlotter.show(mpa.get_norm_acc(), cp_mpa.get_norm_acc(), h_one="Healthy Controls Normalized (n=25)", h_two="CP Patients Normalized (n=12)", alt=True)
        mpa.show_norm_scores(axis=True, include_accuracy=True)
        cp_mpa.show_norm_scores(axis=True, include_accuracy=True)

        # mpa.show_default_scores(axis=True)
        # cp_mpa.show_default_scores(axis=True)

        
        # MatrixPlotter.show(mpa.get_acc(), cp_mpa.get_acc(), h_one="Healthy Controls Default (n=25)", h_two="CP Patients Default (n=12)", alt=True)
        # MatrixPlotter.show(mpa.get_abs_acc(), cp_mpa.get_abs_acc(), h_one="Healthy Controls Absolute Value (n=25)", h_two="CP Patients Absolute Value (n=12)", alt=True)
        

        # mpa.save_shap_values(preds=mpa.get_preds(), title="hc_default")
        # mpa.save_shap_values(preds=mpa.get_abs_preds(), title="hc_abs")
        mpa.save_shap_values(preds=mpa.get_norm_preds(), title="hc_norm")

        # cp_mpa.save_shap_values(preds=cp_mpa.get_preds(), title="cp_default")
        # cp_mpa.save_shap_values(preds=cp_mpa.get_abs_preds(), title="cp_abs")
        cp_mpa.save_shap_values(preds=cp_mpa.get_norm_preds(), title="cp_norm")
        


        print("DONE W INDIV, next combo")



    print("DONE W COMBO, next gs")

    if hcgsmp and cpgsmp:
    #     MatrixPlotter.show(hcgsmp.get_acc(), cpgsmp.get_acc(), h_one="Healthy Controls GS Default (n=25)", h_two="CP Patients GS Default (n=12)", alt=True)
    #     MatrixPlotter.show(hcgsmp.get_abs_acc(), cpgsmp.get_abs_acc(), h_one="Healthy Controls GS Absolute Value (n=25)", h_two="CP Patients GS Absolute Value(n=12)", alt=True)
        MatrixPlotter.show(hcgsmp.get_norm_acc(), cpgsmp.get_norm_acc(), h_one="Healthy Controls GS Normalized (n=25)", h_two="CP Patients GS Normalized (n=12)", alt=True)
        hcgsmp.show_norm_scores(axis=True, include_accuracy=True)
        cpgsmp.show_norm_scores(axis=True, include_accuracy=True)
    
        # hcgsmp.save_shap_values(preds=hcgsmp.get_preds(), title="gs_hc_default")
        # hcgsmp.save_shap_values(preds=hcgsmp.get_abs_preds(), title="gs_hc_abs")
        hcgsmp.save_shap_values(preds=hcgsmp.get_norm_preds(), title="gs_hc_norm")
        # print("DONE W HC BASE")
        # cpgsmp.save_shap_values(preds=cpgsmp.get_preds(), title="gs_cp_default")
        # cpgsmp.save_shap_values(preds=cpgsmp.get_abs_preds(), title="gs_cp_abs")
        cpgsmp.save_shap_values(abs_val=False, non_norm=False, preds=cpgsmp.get_norm_preds(), title="gs_cp_norm_2")
        print("DONE W CP BASE")
        # hc_set_list_preds = hc_set_list.get_all_preds()
        # hc_set_list.save_shap_values(preds=hc_set_list_preds, title="gs_hc_set_list")
        # cp_set_list_preds = cp_set_list.get_all_preds()
        # cp_set_list.save_shap_values(preds=cp_set_list_preds, title="gs_cp_set_list")
    #     print("DONE W COMBO")

    # if hc_combo and cp_combo:
    #     MatrixPlotter.show(hc_combo.get_norm_acc(), cp_combo.get_norm_acc(), h_one="Healthy Controls Combination (n=25)", h_two="CP Patients Combination (n=12)", alt=True)
    # #     print("Done w matrices, on to shap")
    #     hc_combo_preds = hc_combo.get_norm_preds()
    #     hc_combo.save_shap_values(preds=hc_combo_preds, title="hc_combo")

    #     cp_combo_preds = cp_combo.get_norm_preds()
    #     cp_combo.save_shap_values(preds=cp_combo_preds, title="cp_combo")
    # hc_set_list_preds = hc_set_list.get_norm_preds()
    # hc_set_list.save_shap_values(preds=hc_set_list_preds, title="hc_set_list")

    # cp_set_list_preds = cp_set_list.get_norm_preds()
    # cp_set_list.save_shap_values(preds=cp_set_list_preds, title="cp_set_list")


    # MatrixPlotter.show(hcgsmp.get_acc(), cpgsmp.get_acc(), h_one="Healthy Controls GS Default (n=25)", h_two="CP Patients GS Default (n=12)", alt=True)
    # MatrixPlotter.show(hcgsmp.get_abs_acc(), cpgsmp.get_abs_acc(), h_one="Healthy Controls GS Absolute Value (n=25)", h_two="CP Patients GS Absolute Value (n=12)", alt=True)
    # MatrixPlotter.show(hcgsmp.get_norm_acc(), cpgsmp.get_norm_acc(), h_one="Healthy Controls GS Normalized (n=25)", h_two="CP Patients GS Normalized (n=12)", alt=True)


    print("DONE W GS base cases, on to combo")
    import pdb;pdb.set_trace()

    # if hc_set_list and cp_set_list:
    #     MatrixPlotter.show(hc_set_list.get_norm_acc(), cp_set_list.get_norm_acc(), h_one="Healthy Controls GS Combination (n=25)", h_two="CP Patients GS Combination (n=12)", alt=True)
    
    # if hc_combo and hc_set_list:
    #     MatrixPlotter.show(hc_combo.get_norm_acc(), hc_set_list.get_norm_acc(), h_one="Healthy Controls Combination (n=25)", h_two="Healthy Controls GS Combination (n=25)", alt=True)
    
    # if cp_combo and cp_set_list:
    #     MatrixPlotter.show(cp_combo.get_norm_acc(), cp_set_list.get_norm_acc(), h_one="CP Patients Combination (n=12)", h_two="CP Patients GS Combination (n=12)", alt=True)

    print("DONE W Matrices on to shap")

        

    print("DONE W ALL MATRICES AND SHAP")

    import pdb;pdb.set_trace()


# def display_prox_distal_shap_scores(mpa, cp_mpa, hc_combo, cp_combo, hc_set_list, cp_set_list):
    # mpa.show_default_scores(axis=False, models=["RandomForest"])
    # mpa.show_default_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True)
    # cp_mpa.show_default_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True)

    

    # hc_combo.show_norm_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True, use_cat=True)
    # cp_combo.show_norm_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True, use_cat=True)

    
    

    print("DONE W RF COMBO")
    import pdb;pdb.set_trace()

    # hc_set_list.show_all_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True, use_cat=True)
    # cp_set_list.show_norm_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True, use_cat=True)

    print("DONE W ALL RF, on to ExtraTrees")
    import pdb;pdb.set_trace()

    # hc_combo.show_norm_scores(axis=True, models=["ExtraTrees"], include_accuracy=True)
    # cp_combo.show_norm_scores(axis=True, models=["ExtraTrees"], include_accuracy=True)

    # print("DONE W ET COMBO")
    # import pdb;pdb.set_trace()
    
    # hc_set_list.show_norm_scores(axis=True, models=["ExtraTrees"], include_accuracy=True)
    # cp_set_list.show_norm_scores(axis=True, models=["ExtraTrees"], include_accuracy=True)

    print("DONE W ALL ET, finished!!")
    import pdb;pdb.set_trace()



# rpa = MultiPredictor.where(cohort_id=1, task_id=2)[0]
# base case
def gen_mpa_scores(hcgsmp, cpgsmp):
    # hcgsmp.gen_scores_for_mp(force_load=True)
    print("FINISHED WITH HC")
    cpgsmp.gen_scores_for_mp(force_load=True)
    # print("FINISHED WITH CP")

    print("DONE again!")



# mpa.show_predictor_scores(['XGBoost'], reverse_order=True, non_norm=False, abs_val=True, use_cat=True, include_accuracy=True, axis=True)
# rpa.show_predictor_scores(['XGBoost'], reverse_order=True, non_norm=False, abs_val=True, use_cat=True, include_accuracy=True, axis=True)

mpa = MultiPredictor.where(cohort_id=1, task_id=3)[0] # 30 preds, 30 acc
cp_mpa = MultiPredictor.where(cohort_id=2, task_id=3)[0] # 27 preds, 27 acc
# cp_mpa.save_shap_values(abs_val=False, non_norm=False, preds=cp_mpa.get_norm_preds(), title="cp_norm_2")
# mpa.save_shap_values(abs_val=False, non_norm=False, preds=mpa.get_norm_preds(), title="hc_norm_2")

# cp_mpa.show_norm_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True)

# gen_mpa_scores(cp_mpa)


# mpa.gen_scores_for_mp(force_load=True)
# cp_mpa.gen_scores_for_mp(force_load=True)

print("cp stuff done")
# combos

hc_combo = MultiPredictor.where(model="norm_non_abs_combo")[0] # 10 preds
cp_combo = MultiPredictor.get(15) # 9 (norm) preds

# grad sets base case
hcgsmp = MultiPredictor.where(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 1, model="grad_set")[0] # 27 preds, 27 acc
cpgsmp = MultiPredictor.where(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 2, model="grad_set")[0] # 27 preds. 27 acc

hc_set_list = MultiPredictor.where(model="grad_set_combo")[0] # 9 preds, 9 acc
cp_set_list = MultiPredictor.where(model="grad_set_combo", cohort_id=2)[0] # 9 preds, 9 acc

# display_prox_distal_shap_scores(mpa, cp_mpa, hc_combo, cp_combo, hc_set_list, cp_set_list)

# mpa.save_shap_values(abs_val=False, non_norm=False, preds=mpa.get_norm_preds(), title="hc_norm_3")
# cp_mpa.save_shap_values(abs_val=False, non_norm=False, preds=cp_mpa.get_norm_preds(), title="cp_norm_3")

# print("Done w/SG")
# import pdb;pdb.set_trace()


# hcgsmp.save_shap_values(abs_val=False, non_norm=False, preds=hcgsmp.get_norm_preds(), title="gs_hc_norm_3")
# cpgsmp.save_shap_values(abs_val=False, non_norm=False, preds=cpgsmp.get_norm_preds(), title="gs_cp_norm_3")

# print("Done w/GS")
# import pdb;pdb.set_trace()


# cp_mpa.show_norm_scores(axis=True, models=["RandomForest"], include_accuracy=True, use_cat=True)
# mpa.show_norm_scores(models=["RandomForest"], include_accuracy=True, use_cat=True, axis=True)


print("Done w/SG Cluters")


# hcgsmp.show_norm_scores(axis=True, models=["RandomForest"], include_accuracy=True, use_cat=True)
# cpgsmp.show_norm_scores(axis=True, models=["RandomForest"], include_accuracy=True, use_cat=True)

print("Done w/GS Cluters")

cp_combo = MultiPredictor.get(15) # 9 (norm) preds



xgb = [value['XGBoost'] for _, value in mpa.get_norm_acc()]
rfa = [value['RandomForest'] for _, value in mpa.get_norm_acc()]
cba = [value['CatBoost'] for _, value in mpa.get_norm_acc()]
eta = [value['ExtraTrees'] for _, value in mpa.get_norm_acc()]

cp_xgb = [value['XGBoost'] for _, value in cp_mpa.get_norm_acc()]
cp_rfa = [value['RandomForest'] for _, value in cp_mpa.get_norm_acc()]
cp_cba = [value['CatBoost'] for _, value in cp_mpa.get_norm_acc()]
cp_eta = [value['ExtraTrees'] for _, value in cp_mpa.get_norm_acc()]

gs_xgb = [value['XGBoost'] for _, value in hcgsmp.get_norm_acc()]
gs_rfa = [value['RandomForest'] for _, value in hcgsmp.get_norm_acc()]
gs_cba = [value['CatBoost'] for _, value in hcgsmp.get_norm_acc()]
gs_eta = [value['ExtraTrees'] for _, value in hcgsmp.get_norm_acc()]

cp_gs_xgb = [value['XGBoost'] for _, value in cpgsmp.get_norm_acc()]
cp_gs_rfa = [value['RandomForest'] for _, value in cpgsmp.get_norm_acc()]
cp_gs_cba = [value['CatBoost'] for _, value in cpgsmp.get_norm_acc()]
cp_gs_eta = [value['ExtraTrees'] for _, value in cpgsmp.get_norm_acc()]



print("norm ave acc: XGBoost",np.mean(xgb+cp_xgb+gs_xgb+cp_gs_xgb))
print("norm ave acc: RandomForest",np.mean(rfa+cp_rfa+gs_rfa+cp_gs_rfa))
print("norm ave acc: CatBoost",np.mean(cba+cp_cba+gs_cba+cp_gs_cba))
print("norm ave acc: ExtraTrees",np.mean(eta+cp_eta+gs_eta+cp_gs_eta))
print()
print("norm std dev: XGBoost",np.std(xgb+cp_xgb+gs_xgb+cp_gs_xgb))
print("norm std dev: RandomForest",np.std(rfa+cp_rfa+gs_rfa+cp_gs_rfa))
print("norm std dev: CatBoost",np.std(cba+cp_cba+gs_cba+cp_gs_cba))
print("norm std dev: ExtraTrees",np.std(eta+cp_eta+gs_eta+cp_gs_eta))
print()
print("norm median accuracy: XGBoost",np.median(xgb+cp_xgb+gs_xgb+cp_gs_xgb))
print("norm median accuracy: RandomForest",np.median(rfa+cp_rfa+gs_rfa+cp_gs_rfa))
print("norm median accuracy: CatBoost",np.median(cba+cp_cba+gs_cba+cp_gs_cba))
print("norm median accuracy: ExtraTrees",np.median(eta+cp_eta+gs_eta+cp_gs_eta))



mpt = MultiTimePredictor.where(multi_predictor_feature_id=mpa.id, multi_predictor_id=hcgsmp.id)[0]
cp_mpt = MultiTimePredictor.where(multi_predictor_feature_id=cp_mpa.id, multi_predictor_id=cpgsmp.id)[0]
sg_mpt = MultiTimePredictor.where(multi_predictor_feature_id=mpa.id, multi_predictor_id=mpa.id)[0]
cp_sg_mpt = MultiTimePredictor.where(multi_predictor_feature_id=cp_mpa.id, multi_predictor_id=cp_mpa.id)[0]





def display_for_contiuous_paper(mpt, cp_mpt, sg_mpt, cp_sg_mpt):
    MatrixPlotter.show(mpt.get_results(), cp_mpt.get_results(), h_one="Healthy Controls R2", h_two="CP R2", alt=True)
    MatrixPlotter.show(sg_mpt.get_results(), cp_sg_mpt.get_results(), h_one="Healthy Controls R2", h_two="CP R2", alt=True)

    MatrixPlotter.show(mpt.get_mae(), cp_mpt.get_mae(), h_one="HC MAE", h_two="CP MAE")
    MatrixPlotter.show(sg_mpt.get_mae(), cp_sg_mpt.get_mae(), h_one="SG HC MAE", h_two="SG CP MAE")



display_for_contiuous_paper(mpt, cp_mpt, sg_mpt, cp_sg_mpt)
import pdb;pdb.set_trace()



def cont_csv(mpt, sg_mpt, cp_mpt, cp_sg_mpt):
    # display_for_contiuous_paper(mpt, cp_mpt, sg_mpt, cp_sg_mpt)

    import csv
    import json

    # Suppose you have already loaded your four sets of results in these variables:
    #   healthy_controls_full = mpt.get_all_results()     # Example label
    #   healthy_controls_sub  = sg_mpt.get_all_results()  # Example label
    #   cp_full               = cp_mpt.get_all_results()
    #   cp_sub                = cp_sg_mpt.get_all_results()
    #
    # Rename them if you like, or simply plug in the function calls directly.

    healthy_controls_full = mpt.get_all_results()
    healthy_controls_sub  = sg_mpt.get_all_results()
    cp_full               = cp_mpt.get_all_results()
    cp_sub                = cp_sg_mpt.get_all_results()

    # We'll store them all in a list along with a "Dataset" label
    all_datasets = [
        ("HC_FullHand", healthy_controls_full),
        ("HC_SubMotion", healthy_controls_sub),
        ("CP_FullHand", cp_full),
        ("CP_SubMotion", cp_sub),
    ]

    # Create a CSV and write out rows
    with open("combined_results.csv", mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Write header
        writer.writerow([
            "Dataset",
            "Motion",       # e.g. rfin_x, rfrm_x, etc.
            "Regressor",    # e.g. 'AdaBoostRegressor', 'RandomForestRegressor', ...
            "best_cv_mse",
            "best_params",
            "mae",
            "mse",
            "r2"
        ])
        
        # Flatten and write out each record
        for (dataset_label, result_list) in all_datasets:
            # Each element of result_list is like:  ( "rfin_x", { "AdaBoostRegressor": {...}, ... } )
            for (motion_name, model_dict) in result_list:
                # model_dict has keys like 'AdaBoostRegressor', 'CatBoostRegressor', ...
                for regressor_name, metrics in model_dict.items():
                    # metrics is a dict with keys: best_cv_mse, best_params, mae, mse, r2, ...
                    best_cv_mse = metrics["best_cv_mse"]
                    best_params = json.dumps(metrics["best_params"])  # or str(metrics["best_params"])
                    mae         = metrics["mae"]
                    mse         = metrics["mse"]
                    r2          = metrics["r2"]
                    
                    writer.writerow([
                        dataset_label,
                        motion_name,
                        regressor_name,
                        best_cv_mse,
                        best_params,
                        mae,
                        mse,
                        r2
                    ])

    print("Finished writing combined_results.csv!")


import pdb;pdb.set_trace()




# find_or_create_mtp(feature_mp=mpa, timing_mp=hcgsmp)
# find_or_create_mtp(feature_mp=cp_mpa, timing_mp=cpgsmp)
# find_or_create_mtp(feature_mp=cp_mpa, timing_mp=cp_mpa)



# find_or_create_mtp(feature_mp=mpa, timing_mp=mpa)



import pdb;pdb.set_trace()


# display_for_paper(mpa=mpa, cp_mpa=cp_mpa, hc_combo=hc_combo, cp_combo=cp_combo, hcgsmp=hcgsmp, cpgsmp=cpgsmp, hc_set_list=hc_set_list, cp_set_list=cp_set_list)
# cp_combo.show_norm_scores(axis=True, models=["RandomForest", "XGBoost", "ExtraTrees", "CatBoost"], include_accuracy=True)

# gen_mpa_scores(hcgsmp, cpgsmp)

print("REALLY DONE")


# cpgsmp.get_norm_preds()[0].get_df()
# cpgsmp.gen_scores_for_mp(force_load=True)

print('wow done')
# add_set_stats()

# hcgsmp.gen_scores_for_mp(force_load=True)
# cpgsmp.gen_scores_for_mp(force_load=True)

print("wow done")

# 101 => cp s002
# 97 = S015




# rtt = Task.get(3).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="rfhd_x")[0])
# ltt = Task.get(3).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="rwrb_x")[0])c

# nd_rtt = Task.get(4).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lfhd_x")[0])
# nd_ltt = Task.get(4).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lwrb_x")[0])

# cp_rtt = Task.get(3).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="rfhd_x")[0], cohort=Cohort.get(2))
# cp_ltt = Task.get(3).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="rwrb_x")[0], cohort=Cohort.get(2))

# nd_cp_rtt = Task.get(4).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lfhd_x")[0], cohort=Cohort.get(2))
# nd_cp_ltt = Task.get(4).full_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lwrb_x")[0], cohort=Cohort.get(2))

# nd_rtt = Task.get(4).set_combined_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lfhd_x")[0])
# nd_ltt = Task.get(4).set_combined_gradient_set_stats_by_task_and_cohort(Sensor.where(name="lwrb_x")[0])



SENSOR_CODES = [
    'rfin_x',
    'rwra_x',
    'rwrb_x',
    'rfrm_x',
    'relb_x',
    # 'relbm_x',
    'rupa_x',
    'rsho_x',
    'rbhd_x',
    'rfhd_x',
]




def compare_and_correlate_features(df1, df2):
    """
    Compare features from two dataframes and show their relative correlation.
    
    :param df1: First dataframe
    :param df2: Second dataframe
    """
    # Get common features
    common_features = list(set(df1.columns) & set(df2.columns))
    
    print(f"Number of features in df1: {len(df1.columns)}")
    print(f"Number of features in df2: {len(df2.columns)}")
    print(f"Number of common features: {len(common_features)}")
    
    # Print unique features
    print("\nUnique features in df1:")
    print(list(set(df1.columns) - set(common_features)))
    print("\nUnique features in df2:")
    print(list(set(df2.columns) - set(common_features)))
    
    # Compute correlation matrix for common features
    df1_common = df1[common_features]
    df2_common = df2[common_features]
    
    # Combine the common features from both dataframes
    combined_df = pd.concat([df1_common, df2_common], axis=0)
    
    # Compute the correlation matrix
    correlation_matrix = combined_df.corr()
    
    # Plot heatmap of correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title("Correlation Heatmap of Common Features")
    plt.tight_layout()
    plt.show()
    
    # Return correlation matrix for further analysis if needed
    return correlation_matrix

def boruta_feature_selection(df, target_column=None, n_estimators=100, max_iter=100):
    """
    Apply Boruta feature selection to a dataframe.
    
    :param df: Input dataframe
    :param target_column: Name of the target column. If None, use the last column as target.
    :param n_estimators: Number of estimators for the Random Forest
    :param max_iter: Maximum number of iterations for Boruta
    :return: Dataframe with selected features
    """
    if target_column is None:
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    else:
        X = df.drop(columns=[target_column])
        y = df[target_column]
    
    # Initialize Boruta
    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=-1, max_depth=5)
    boruta = BorutaPy(rf, n_estimators=n_estimators, max_iter=max_iter, verbose=2)
    
    # Fit Boruta
    boruta.fit(X.values, y.values)
    
    # Get selected feature names
    selected_features = X.columns[boruta.support_].tolist()
    
    if target_column is not None:
        selected_features.append(target_column)
    
    # Return dataframe with selected features
    return df[selected_features]

def new_compare_tsfresh_dataframes(combined_df1_cleaned, combined_df2_cleaned, title=None):
    """
    Compare two tsfresh feature dataframes by calculating the correlation between their features
    and displaying the results as a heatmap of relative similarities, with sensors ordered as specified.
    
    Parameters:
    combined_df1_cleaned (pd.DataFrame): Cleaned DataFrame for the first set of sensors.
    combined_df2_cleaned (pd.DataFrame): Cleaned DataFrame for the second set of sensors.
    title (str, optional): Title for the heatmap plot.
    
    Returns:
    pd.DataFrame: The relative similarity matrix between the sensors.
    """
    
    SENSOR_ORDER = [
        'fin',
        'wra',
        'wrb',
        'frm',
        'elb',
        'upa',
        'sho',
        'bhd',
        'fhd',
    ]
    
    def get_sensor_names(df):
        return sorted(list(set([col.split('_')[0] for col in df.columns])), 
                      key=lambda x: SENSOR_ORDER.index(x[1:]) if x[1:] in SENSOR_ORDER else len(SENSOR_ORDER))
    
    def get_sensor_data(df, sensor):
        return df[[col for col in df.columns if col.startswith(sensor)]]
    
    from scipy.spatial.distance import euclidean, cosine
    from scipy import stats

    def comprehensive_dataframe_comparison(df1, df2):
        """
        Perform a comprehensive comparison between two dataframes of sensor data.
        
        :param df1: First dataframe
        :param df2: Second dataframe
        """
        # Ensure both dataframes have the same features
        common_features = list(set(df1.index) & set(df2.index))
        df1 = df1.loc[common_features]
        df2 = df2.loc[common_features]
        
        # 1. Pearson Correlation
        pearson_corr = stats.pearsonr(df1, df2)[0]
        
        # 2. Spearman Rank Correlation
        spearman_corr = stats.spearmanr(df1, df2)[0]
        
        # 3. Kendall's Tau Correlation
        kendall_tau = stats.kendalltau(df1, df2)[0]
        
        # 4. Euclidean Distance
        euclidean_dist = euclidean(df1, df2)
        
        # 5. Cosine Distance
        cosine_dist = cosine(df1, df2)
        
        # 6. Feature-by-feature comparison
        feature_diff = pd.DataFrame({
            'df1': df1,
            'df2': df2,
            'abs_diff': np.abs(df1 - df2),
            'percent_diff': np.abs((df1 - df2) / df1) * 100
        })
        
        # 7. Distribution comparison using Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(df1, df2)
        
        # Print results
        print(f"Pearson Correlation: {pearson_corr:.4f}")
        print(f"Spearman Rank Correlation: {spearman_corr:.4f}")
        print(f"Kendall's Tau: {kendall_tau:.4f}")
        print(f"Euclidean Distance: {euclidean_dist:.4f}")
        print(f"Cosine Distance: {cosine_dist:.4f}")
        print(f"Kolmogorov-Smirnov Test Statistic: {ks_statistic:.4f}")
        print(f"Kolmogorov-Smirnov p-value: {ks_pvalue:.4f}")
    
    def sensor_similarity(sensor1_df, sensor2_df):
        """
        Computes the average Euclidean distance across x, y, z axes between two sensors.
        
        Parameters:
        - sensor1_df (pd.DataFrame): DataFrame containing data for sensor1.
        - sensor2_df (pd.DataFrame): DataFrame containing data for sensor2.
        
        Returns:
        - float: The average Euclidean distance across the three axes.
        """
        axes = ['x', 'y', 'z']
        distances = []
        for axis in axes:
            # Corrected the matching to look for columns ending with '_x', '_y', '_z'
            s1_col = [col for col in sensor1_df.columns if col.endswith(f'_{axis}')]
            s2_col = [col for col in sensor2_df.columns if col.endswith(f'_{axis}')]
            if s1_col and s2_col:
                try:
                    # Assuming one column per axis per sensor
                    dist = euclidean(sensor1_df[s1_col[0]], sensor2_df[s2_col[0]])
                    if np.isfinite(dist):
                        distances.append(dist)
                except Exception as e:
                    print(f"Error calculating Euclidean distance for {s1_col[0]} and {s2_col[0]}: {e}")
        return np.mean(distances) if distances else np.nan


    sensors1 = get_sensor_names(combined_df1_cleaned)
    sensors2 = get_sensor_names(combined_df2_cleaned)

    similarity_matrix = pd.DataFrame(index=sensors1, columns=sensors2)
    for sensor1 in sensors1:
        sensor1_df = get_sensor_data(combined_df1_cleaned, sensor1)
        for sensor2 in sensors2:
            sensor2_df = get_sensor_data(combined_df2_cleaned, sensor2)
            similarity = sensor_similarity(sensor1_df, sensor2_df)
            similarity_matrix.loc[sensor1, sensor2] = similarity

    # Replace any remaining NaN values with the minimum similarity for normalization

    min_similarity = similarity_matrix.min().min()
    similarity_matrix = similarity_matrix.fillna(min_similarity)

    # Normalize similarities to highlight relative differences
    normalized_matrix = (similarity_matrix - similarity_matrix.min().min()) / (similarity_matrix.max().max() - similarity_matrix.min().min())

    # Create a heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(normalized_matrix, annot=similarity_matrix.round(4), cmap='YlGnBu', vmin=0, vmax=1, fmt='.4f')
    plt.title(title or 'Relative Sensor Similarity Heatmap')
    plt.xlabel('Left Side Sensors')
    plt.ylabel('Right Side Sensors')
    plt.tight_layout()
    
    if title:
        # Create a 'results' directory at the same level as the current directory if it doesn't exist
        results_dir = os.path.join(os.path.dirname(os.getcwd()), 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate a valid filename
        filename_base = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in title)
        
        # Save heatmap
        heatmap_path = os.path.join(results_dir, f'{filename_base}_heatmap.png')
        plt.savefig(heatmap_path)
        plt.show()
        
        # Save original similarity matrix
        import pdb;pdb.set_trace()
        similarity_path = os.path.join(results_dir, f'{filename_base}_similarity_matrix.csv')
        similarity_matrix.to_csv(similarity_path)
        
        # Save normalized similarity matrix
        normalized_path = os.path.join(results_dir, f'{filename_base}_normalized_matrix.csv')
        normalized_matrix.to_csv(normalized_path)
        
        print(f"Results saved in '{results_dir}' directory:")
        print(f"- Heatmap: {heatmap_path}")
        print(f"- Similarity Matrix: {similarity_path}")
        print(f"- Normalized Matrix: {normalized_path}")
    

    return similarity_matrix, normalized_matrix

def compare_tsfresh_dataframes(dataframes_list1, dataframes_list2, title=None):
    def compare_index_suffixes(df1, df2, name1, name2, position):
        suffixes1 = [col.split('__', 1)[1] if '__' in col else col for col in df1.index]
        suffixes2 = [col.split('__', 1)[1] if '__' in col else col for col in df2.index]
        
        if suffixes1 != suffixes2:
            mismatched = [f"{s1} != {s2}" for s1, s2 in zip(suffixes1, suffixes2) if s1 != s2]
            mismatch_msg = ', '.join(mismatched[:5])  # Limit to first 5 mismatches for brevity
            if len(mismatched) > 5:
                mismatch_msg += f", and {len(mismatched) - 5} more"
            return False, f"Dataframes '{name1}' and '{name2}' at position {position} have mismatched features after prefix: {mismatch_msg}"
        return True, ""
    """
    Compare two lists of tsfresh feature dataframes by calculating the correlation between their features
    and displaying the results as a heatmap.

    Parameters:
    dataframes_list1 (list of tuples): List of tuples containing DataFrames and sensor names for the first set.
    dataframes_list2 (list of tuples): List of tuples containing DataFrames and sensor names for the second set.

    Returns:
    pd.DataFrame: The correlation matrix between the mean columns of the dataframes.
    """
    
    # Fill NaN values with 0 or any other appropriate method
    def filter_rows(df1, df2):
        """
        Removes rows from both DataFrames where all values in the corresponding rows of both DataFrames are either 0 or NaN.

        Parameters:
        - df1: First DataFrame.
        - df2: Second DataFrame.

        Returns:
        - A tuple containing the filtered versions of df1 and df2.
        """
        # Check if both DataFrames have the same index
        if not df1.index.equals(df2.index):
            raise ValueError("Both DataFrames must have the same index.")

        # Fill NaN values with 0
        df1_filled = df1.fillna(0)
        df2_filled = df2.fillna(0)

        # Create boolean masks where each element is True if it's 0
        mask_df1 = df1_filled == 0
        mask_df2 = df2_filled == 0

        # Identify rows where all values in df1 are 0
        rows_all_zero_df1 = mask_df1.all(axis=1)

        # Identify rows where all values in df2 are 0
        rows_all_zero_df2 = mask_df2.all(axis=1)

        # Rows to drop: Rows where both df1 and df2 have all values as 0
        rows_to_drop = rows_all_zero_df1 & rows_all_zero_df2

        # Filter out these rows from both DataFrames
        df1_filtered = df1_filled[~rows_to_drop].copy()
        df2_filtered = df2_filled[~rows_to_drop].copy()

        return df1_filtered, df2_filtered

    # Example Usage:
    # Assuming dataframes_list1 and dataframes_list2 are your DataFrames

    filtered_df1, filtered_df2 = filter_rows(dataframes_list1, dataframes_list2)
    correlation_matrix = new_compare_tsfresh_dataframes(filtered_df1, filtered_df2, title)
    print(correlation_matrix)


def construct_corr(tsk=None, cohort=None, title=None, use_dom=True, sub_stat=False, versus_opposite_side=False):
    arr_rtt = []
    arr_ltt = []

    if not tsk:
        tsk = Task.get(3)

    if use_dom:
        main_task = Task.get(3)
        compare_task = Task.get(3)
    else:
        main_task = Task.get(4)
        compare_task = Task.get(4)

    for code in SENSOR_CODES:
        sensor = Sensor.where(name=code)[0]
        alt_side_sensor = Sensor.where(name=Task.get_counterpart_sensor(sensor.name))[0]

        if not use_dom and not versus_opposite_side:
            sensor = alt_side_sensor
            alt_side_sensor = sensor

        if versus_opposite_side:
            compare_sensor = alt_side_sensor
        else:
            compare_sensor = sensor

        sen_set = sensor.get_set()
        compare_sen_set = compare_sensor.get_set()

        for sen, compare_sen in zip(sen_set, compare_sen_set):
            rtt = main_task.full_gradient_set_stats_by_task_and_cohort(sen, cohort=cohort, get_agg_sub_stat=sub_stat)
            ltt = compare_task.full_gradient_set_stats_by_task_and_cohort(compare_sen, cohort=cohort, get_agg_sub_stat=sub_stat)


            arr_rtt.append((rtt, sen.name))
            arr_ltt.append((ltt, compare_sen.name))
        
    def make_dfs(arr):
        dfs = [item[0] for item in arr]
        labels = [item[1] for item in arr]
        # Concatenate along the rows (axis=0)
        combined_df = pd.concat(dfs, axis=0, ignore_index=True)

        # Assign labels as the index
        combined_df['label'] = labels
        combined_df.set_index('label', inplace=True)
        
        return combined_df
    alt_arr_ltt = arr_ltt
    import pdb;pdb.set_trace()
    arr_ltt = make_dfs(arr_ltt).T
    arr_rtt = make_dfs(arr_rtt).T

    def filter_dataframes(df_left: pd.DataFrame, df_right: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Ensure both DataFrames have the same index
        if not df_left.index.equals(df_right.index):
            import pdb;pdb.set_trace()

            raise ValueError("Both DataFrames must have the same index.")

        # Create boolean masks where each element is True if it's 0 or NaN
        mask_left = df_left.eq(0) | df_left.isna()
        mask_right = df_right.eq(0) | df_right.isna()

        # For each row, check if all elements satisfy the condition (0 or NaN)
        rows_all_zero_or_nan_left = mask_left.all(axis=1)
        rows_all_zero_or_nan_right = mask_right.all(axis=1)

        # Identify rows where both DataFrames have all elements as 0 or NaN
        rows_to_drop = rows_all_zero_or_nan_left & rows_all_zero_or_nan_right

        # Filter out these rows from both DataFrames
        filtered_df_left = df_left[~rows_to_drop].copy()
        filtered_df_right = df_right[~rows_to_drop].copy()
        return filtered_df_left, filtered_df_right
    

    arr_ltt, arr_rtt = filter_dataframes(arr_ltt, arr_rtt)

    compare_tsfresh_dataframes(arr_rtt, arr_ltt, title=title)


hc = Cohort.get(1)
cp = Cohort.get(2)





# rtt = PatientTask.where(id=97)[0].combined_sub_gradient_stats_list(Sensor.where(name="rfhd_x")[0], non_normed=False, loc=False)
# ltt = PatientTask.where(id=97)[0].combined_sub_gradient_stats_list(Sensor.where(name="rwrb_x")[0], non_normed=False, loc=False)

# cp_rtt = PatientTask.where(id=97)[0].combined_sub_gradient_stats_list(Sensor.where(name="rfhd_x")[0], non_normed=False, loc=False)
# cp_ltt = PatientTask.where(id=97)[0].combined_sub_gradient_stats_list(Sensor.where(name="rwrb_x")[0], non_normed=False, loc=False)



# grad set combo



# cp_mpa.gen_scores_for_mp(force_load=True)
print("cp new scores")
# make_final_combo(cp_mpa)

print("Final combo made")




# display_for_paper(mpa=mpa, cp_mpa=cp_mpa, hcgsmp=hcgsmp, hc_combo=hc_combo, cp_combo=cp_combo, cpgsmp=cpgsmp, hc_set_list=hc_set_list, cp_set_list=cp_set_list)

print("done displaying for paper")

# display_prox_distal_shap_scores(hc_combo, cp_combo, hc_set_list, cp_set_list)

# todo          
# display_prox_distal_shap_scores(hc_combo, cp_combo, hc_set_list, cp_set_list)

# done 7.24
# display_for_paper(mpa, cp_mpa, hc_combo, cp_combo, hcgsmp, cpgsmp, hc_set_list, cp_set_list)


print("Done w paper display")





bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
sp = MultiPredictor.get(15)
# MatrixPlotter.show(bc.get_all_acc(), sp.get_all_acc(), h_one="Healthy Controls (n=25)", h_two="CP Patients (n=7)", alt=True)

hc_bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
hc_combo = MultiPredictor.get(15)
hc_set_list = MultiPredictor.where(model="grad_set_combo")[0]


cp_bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
cp_combo = MultiPredictor.get(15)
hc_set_list = MultiPredictor.where(model="grad_set_combo")[0]


bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
mp = MultiPredictor.where(cohort_id=2)[0]

combo_cp = MultiPredictor.get(15)

hcgsmp = MultiPredictor.where(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 1, model="grad_set")[0]
cpgsmp = MultiPredictor.where(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 2, model="grad_set")[0]


# bc.get_norm_corr()
# no driving feature, combination more important
# add to google doc supp, put word in google doc
# add to methods splitting folds
# discussion short paragraph summarizing result, dont go into too much detail
# limitations
# future directions
# conclusion
# put references in a comment
# cp_mpa = MultiPredictor.where(cohort_id=2, task_id=3)[0] # 27 preds, 27 acc
# preds = cp_mpa.get_norm_corr()
# import pdb;pdb.set_trace()
# for pred in preds:
#     pred.train_from(force_load=True)

# import pdb;pdb.set_trace()
# for ps in PredictorScore.where(classifier_name="RandomForest", predictor_id=bc.get_norm_preds()[1].id):
#     ps.view_shap_plot(show_plot=True, show_fold_corr=True)




# try_getting_all_shap(bc)
data = {
    'Subject Number': [1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
    'Tightness': [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0],
    'Embarrassment or shame': [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
    'Temperature': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
    'Communication with healthcare': [1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
    'Basic Movement Restriction': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
    'Comfort': [0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1],
    'Sleeping with the brace': [1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0],
    'Fear regarding Scoliosis/Bracing': [0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1],
    'Peer reaction': [0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1],
    'Pain': [1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    'Participation in sports, activities, and exercise': [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1],
    'Non compliance due to Pain/Discomfort': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    'Functional Adaptations': [1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    'While Sitting': [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'Impact on self-esteem & personal identity': [0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'Skin Dryness/Irritation': [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    'While active': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    'Non-compliance due to Functional impediment': [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    'Family support': [0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    'SRS Function': [5, 3.2, 4, 2.2, 4.8, 4.6, 3.6, 4.8, 4.4, 5, 3.4, 4.2, 4.2, 3.8, 4.8, 4],
    'SRS Pain': [4.8, 3.4, 5, 3, 5, 4.6, 4.8, 4, 4.2, 5, 3.8, 5, 4.4, 3.4, 5, 4.4],
    'SRS Self-Image': [4.6, 2.4, 4, 2.8, 2.8, 4, 4.2, 3.8, 3.4, 5, 1.8, 4.2, 2.4, 3.4, 3.6, 3.6],
    'SRS Mental Health': [5, 2.8, 3.2, 2.4, 3.8, 5, 3.8, 3.2, 4.4, 5, 3.4, 4.8, 3.6, 3.8, 3, 4.4],
    'SRS Satisfaction with Management': [5, 3.5, 2.5, 2, 1.5, 5, 4.5, 3.5, 4, 5, 4, 3.5, 4, 4, 2, 3.5]
}


df = pd.DataFrame(data)

complaints = [
    'Tightness',
    'Embarrassment or shame',
    'Temperature',
    'Communication with healthcare',
    'Basic Movement Restriction',
    'Comfort',
    'Sleeping with the brace',
    'Fear regarding Scoliosis/Bracing',
    'Peer reaction',
    'Pain',
    'Participation in sports, activities, and exercise',
    'Non compliance due to Pain/Discomfort',
    'Functional Adaptations',
    'While Sitting',
    'Impact on self-esteem & personal identity',
    'Skin Dryness/Irritation',
    'While active',
    'Non-compliance due to Functional impediment',
    'Family support'
]

srs_scores = [
    'SRS Function',
    'SRS Pain',
    'SRS Self-Image',
    'SRS Mental Health',
    'SRS Satisfaction with Management'
]



# Initialize an empty DataFrame to store correlation coefficients
corr_matrix = pd.DataFrame(index=complaints, columns=srs_scores)

import shap
fetch_new_tasks()
def random_forest_analysis_with_shap(target_variable):
    X = df[complaints]
    y = df[target_variable]
    
    # Initialize the model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    rf.fit(X, y)
    
    # Cross-validation to estimate performance
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)
    print(f'RMSE for {target_variable}: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}')
    
    # Compute SHAP values
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)
    
    # Plot SHAP summary plot (bar chart)
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f'SHAP Feature Importances for Predicting {target_variable}')
    plt.tight_layout()
    plt.show()


# Perform analysis for each SRS score
# for score in srs_scores:
#     print(f'Random Forest Analysis for {score}')
#     random_forest_analysis_with_shap(score)

# Features: SRS scores
X = df[srs_scores]

# Target variable: 'Temperature' complaint
y = df['Temperature']

# Check the distribution of the target variable
print("Distribution of 'Temperature' Complaint:")
print(y.value_counts())

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation predictions
y_pred = cross_val_predict(rf_classifier, X, y, cv=5)

# Classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# Fit the model on the entire dataset to get feature importances
rf_classifier.fit(X, y)
importances = rf_classifier.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'SRS Score': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='SRS Score', data=importance_df)
plt.title("Feature Importances for Predicting 'Temperature' Complaint")
plt.xlabel('Importance')
plt.ylabel('SRS Score')
plt.tight_layout()
plt.show()


# Features: SRS scores
X = df[['SRS Function', 'SRS Pain', 'SRS Self-Image', 'SRS Mental Health', 'SRS Satisfaction with Management']]

# Target variable: 'Temperature' complaint
y = df['Temperature']

print("Distribution of 'Temperature' Complaint:")
print(y.value_counts())

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

y_pred = cross_val_predict(rf_classifier, X, y, cv=5)

# Classification report
print("Classification Report:")
print(classification_report(y, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y, y_pred))

# Fit the model on the entire dataset to get feature importances
rf_classifier.fit(X, y)
importances = rf_classifier.feature_importances_

# Create a DataFrame for visualization
importance_df = pd.DataFrame({
    'SRS Score': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='SRS Score', data=importance_df)
plt.title("Feature Importances for Predicting 'Temperature' Complaint")
plt.xlabel('Importance')
plt.ylabel('SRS Score')
plt.tight_layout()
plt.show()

from scipy import stats

# Calculate the point-biserial correlation for each complaint and SRS score pair
for complaint in complaints:
    for score in srs_scores:
        corr, p_value = stats.pointbiserialr(df[complaint], df[score])
        corr_matrix.loc[complaint, score] = corr

corr_matrix = corr_matrix.astype(float)

plt.figure(figsize=(12, 10))

# Create a heatmap with annotations
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")

# Set the title
plt.title('Correlation Heatmap between Complaints and SRS Scores')

# Adjust the layout
plt.tight_layout()

# Display the heatmap
plt.show()


# construct_corr(cohort=hc, use_dom=True, versus_opposite_side=True, sub_stat=True, title=" SUB STAT Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=True, versus_opposite_side=True, sub_stat=True, title=" SUB STAT  Dom CP")

# construct_corr(cohort=hc, use_dom=False, versus_opposite_side=True, sub_stat=True, title=" SUB STAT  Non Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=False, versus_opposite_side=True, sub_stat=True, title=" SUB STAT  Non Dom CP")

# construct_corr(cohort=hc, use_dom=True, versus_opposite_side=False, sub_stat=True, title=" SUB STAT Same Arm Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=True, versus_opposite_side=False, sub_stat=True, title=" SUB STAT Same Arm Dom CP")

# construct_corr(cohort=hc, use_dom=False, versus_opposite_side=False, sub_stat=True, title=" SUB STAT Same Arm Non Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=False, versus_opposite_side=False, sub_stat=True, title=" SUB STAT Same Arm Non Dom CP")

# construct_corr(cohort=hc, use_dom=True, versus_opposite_side=True, sub_stat=False, title="Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=True, versus_opposite_side=True, sub_stat=False, title="Dom CP")

# construct_corr(cohort=hc, use_dom=False, versus_opposite_side=True, sub_stat=False, title="Non Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=False, versus_opposite_side=True, sub_stat=False, title="Non Dom CP")

# construct_corr(cohort=hc, use_dom=True, versus_opposite_side=False, sub_stat=False, title="Same Arm Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=True, versus_opposite_side=False, sub_stat=False, title="Same Arm Dom CP")

# construct_corr(cohort=hc, use_dom=False, versus_opposite_side=False, sub_stat=False, title="Same Arm Non Dom Healthy Controls")
# construct_corr(cohort=cp, use_dom=False, versus_opposite_side=False, sub_stat=False, title="Same Arm Non Dom CP")




## NON COMBO


## COMBO

# mp.view_progress(fix_missing_acc=True)

# make_final_combo(cpgsmp)

import pdb;pdb.set_trace()

# mpa.show_predictor_scores()

cpgsmp.gen_scores_for_mp()



# started first
# mp.gen_scores_for_mp(force_load=True)

# sp = MultiPredictor.get(15)

import pdb;pdb.set_trace()



# mp.gen_scores_for_mp(force_load=True)


# make_final_cp_combo(mp, sp)




cpd = MultiPredictor.find_or_create(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 2, model="default")
cpd.gen_scores_for_mp(force_load=True)


print("DONE W NEW CP SCORES")
import pdb;pdb.set_trace()


hcgsmp = MultiPredictor.find_or_create(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 2, model="grad_set")

print("DONE W HEALTHY COHORT")

import pdb;pdb.set_trace()


hcgsmp = MultiPredictor.where(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 2, model="grad_set")[0]
cpgsmp = MultiPredictor.find_or_create(task_id = Task.where(description="Block_dominant")[0].id, cohort_id = 1, model="grad_set")


# add_set_stats()    
mp = MultiPredictor.where(cohort_id=2)[0]
sp = MultiPredictor.get(15)
import pdb;pdb.set_trace()

mp.view_progress()




import pdb;pdb.set_trace()

print("OK DONE")
sp = MultiPredictor.get(15)
pred = sp.get_all_preds()[2]
df = pred.get_df()




rpa = MultiPredictor.where(cohort_id=1, task_id=2)[0]
pred = rpa.get_preds()[1]
pred.get_df()

def automate_construct_corr():
    cohorts = [('hc', 'Healthy Controls'), ('cp', 'CP')]
    use_dom_options = [(True, 'Dom'), (False, 'Non Dom')]
    versus_opposite_side_options = [(True, 'versus_opposite_side'), (False, 'Same Arm')]
    sub_stat_options = [(True, 'SUB STAT'), (False, '')]

    for cohort, cohort_name in cohorts:
        for use_dom, dom_desc in use_dom_options:
            for versus_opposite_side, side_desc in versus_opposite_side_options:
                for sub_stat, sub_stat_desc in sub_stat_options:
                    title = f"{sub_stat_desc} {dom_desc} {side_desc} {cohort_name}".strip()
                    construct_corr(cohort=cohort, use_dom=use_dom, versus_opposite_side=versus_opposite_side, sub_stat=sub_stat, title=title)



cp_cohort = Cohort.where(name=COHORT_NAME)[0]
block_task = Task.find_by("description", TASK_NAME)
# mp = MultiPredictor.where(task_id = block_task.id, cohort_id = cp_cohort.id)[0]


save_sp_shap_vals(sp)

def calculate_statistics(values):
    if not values:
        print("No values to calculate statistics.")
        return None, None, None
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    std_dev = variance ** 0.5
    return mean, variance, std_dev

# First set of predictions

def compare_motion():
    test_list = []
    for pr in bc.get_all_preds():
        num_features = len(pr.get_predictor_scores()[0].get_top_n_features(500))
        test_list.append(num_features)
        print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, Sensor.get(pr.sensor_id).name, "num features:", num_features)

    # Calculate and print statistics for the first set
    mean, variance, std_dev = calculate_statistics(test_list)
    if mean is not None:
        print("First Set - MEAN:", mean, "VARIANCE:", variance, "STD DEV:", std_dev)

    # Second set of predictions
    test_list = []
    for pr in sp.get_all_preds():
        ps = pr.get_predictor_scores()
        if len(ps) != 0:
            num_features = len(ps[0].get_top_n_features(500))
            test_list.append(num_features)
            print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, Sensor.get(pr.sensor_id).name, "num features:", num_features)
        else:
            print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, Sensor.get(pr.sensor_id).name, "SKIPPED")

    # Calculate and print statistics for the second set
    mean, variance, std_dev = calculate_statistics(test_list)
    if mean is not None:
        print("Second Set - MEAN:", mean, "VARIANCE:", variance, "STD DEV:", std_dev)





bc.show_norm_scores(axis=True, include_accuracy=True)
import pdb;pdb.set_trace()
# sp.show_norm_scores(axis=True, include_accuracy=True)



import pdb;pdb.set_trace()

# generate_current_cp_scores(mp)


import pdb;pdb.set_trace()
print("Done!! Yolo")



bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
mp = MultiPredictor.get(15)


pss = PredictorScore.where(multi_predictor_id=15)

for ps in pss:
    ps.view_shap_plot(title="CP")



import pdb;pdb.set_trace()
# MatrixPlotter.show(bc.get_norm_acc(), mp.get_norm_acc(), h_one="Block Combo (n = 24)", h_two="CP Block Combo (n=6)")


# mp.show_
import pdb;pdb.set_trace()
# mpc.aggregate_shap_values(non_norm=False)
# feature_cluster_map = mpc.feature_cluster_map(non_norm=False)





pr = Predictor.where(multi_predictor_id=6)
# healthy controls block
mpa = MultiPredictor.where(cohort_id=1, task_id=3)[0]

# cp block
mpb = MultiPredictor.where(cohort_id=1, task_id=3)[1]

# combos block hc
mpc = MultiPredictor.where(cohort_id=1, task_id=3)[2]

# ring tasks
rpa = MultiPredictor.where(cohort_id=1, task_id=2)[0]

# combo ring preds
rpb = MultiPredictor.where(cohort_id=1, task_id=2)[1]
# mpa.save_shap_values(abs_val=False, non_norm=False)
# block combo
bc = MultiPredictor.where(model="norm_non_abs_combo")[0]

# ring combo
rc = MultiPredictor.where(model="norm_non_abs_combo")[1]

import pdb;pdb.set_trace()

def fix_pd_et_cohort_ids():
    g1 = ['amg__S008','amg__S009','amg__S011','amg__S015','amg__S016','amg__S019','amg__S022','amg__S027','amg__S028','amg__S029','amg__S030','amg__S031',]
    c1 = 'group_1_analysis_me'
    pc1 = Cohort.where(name=c1)[0]
    pg1 = Patient.where(name=g1)
    for pt in pg1:
        pt.update(cohort_id=pc1.id)
        ptts = PatientTask.where(patient_id=pt.id)
        for ptt in ptts:
            ptt.update(cohort_id=pc1.id)
            ptt.save()
        pt.save()

    g2 = ['amg__S010','amg__S013','amg__S014','amg__S017','amg__S018','amg__S021','amg__S023','amg__S024','amg__S025','amg__S026',]
    c2 = 'group_2_analysis_me'
    pc2 = Cohort.where(name=c2)[0]
    pg2 = Patient.where(name=g2)
    for pt in pg2:
        pt.update(cohort_id=pc2.id)
        ptts = PatientTask.where(patient_id=pt.id)
        for ptt in ptts:
            ptt.update(cohort_id=pc2.id)
            ptt.save()
        pt.save()
    g3 = ['amg__S003','amg__S007','amg__S012','amg__S020',]
    c3 = 'group_3_analysis_me'
    pc3 = Cohort.where(name=c3)[0]
    pg3 = Patient.where(name=g3)
    for pt in pg3:
        pt.update(cohort_id=pc3.id)
        ptts = PatientTask.where(patient_id=pt.id)
        for ptt in ptts:
            ptt.update(cohort_id=pc3.id)
            ptt.save()
        pt.save()


def fix_cp_cohort_ids():
    g1 = ['S008_cp', 'S002_cp', 'S001_cp', 'S006_cp', 'S003_cp']
    c1 = 'cp_before'
    pc1 = Cohort.where(name=c1)[0]
    pg1 = Patient.where(name=g1)
    for pt in pg1:
        pt.update(cohort_id=pc1.id)
        ptts = PatientTask.where(patient_id=pt.id)
        for ptt in ptts:
            print(pc1.id)
            ptt.update(cohort_id=pc1.id)



import pdb;pdb.set_trace()
SigCheck().compare_obj(rc, bc)

# Does the same thing as in your console. See top of page for importing 
# rc.show_norm_scores(axis=True, include_accuracy=True)


# bc.show_norm_scores(axis=True, include_accuracy=True)



# pprint.pp(bc.get_new_axis(abs_val=False, non_norm=False))

# MatrixPlotter.show(mpa.get_acc(), rpa.get_acc(), alt=True, h_one="Block", h_two="Ring")

# MatrixPlotter.show(mpa.get_abs_acc(), rpa.get_abs_acc(), alt=True, h_one="Block Absolute Values", h_two="Ring Absolute Values")

# MatrixPlotter.show(mpa.get_norm_acc(), rpa.get_norm_acc(), alt=True, h_one="Block Normalized", h_two="Ring Normalized")
# mpa.get_all_preds()[-1].train_from(get_sg_count=True)
# bc = MultiPredictor.get(7)
# rc = MultiPredictor.get(8)




def show_shap_stats(multi_preds, combo=False):
    def print_data(pr, test_list):
        num_features = len(pr.get_predictor_scores()[0].get_top_n_features(500))
        test_list.append(num_features)
        print(Sensor.get(pr.sensor_id).name, "num features:", num_features)
        return test_list
    
    def display_sd(test_list):
        print("The original list : " + str(test_list)) 

        # Printing result
        mean = sum(test_list) / len(test_list) 
        variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
        res = variance ** 0.5
        print("Mean", mean)
        print("Standard deviation of sample is : " + str(res)) 
        return []

    if combo is False:
        for mp in multi_preds:
            print(Task.get(mp.task_id).description, "Default")
            test_list = []
            for pr in mp.get_preds():
                tl = print_data(pr, test_list)
            test_list = display_sd(test_list)

            print()
            print(Task.get(mp.task_id).description, "ABS")
            for pr in mp.get_abs_preds():
                tl = print_data(pr, test_list)
            test_list = display_sd(test_list)
    
            print()
            print(Task.get(mp.task_id).description, "Norm")
            for pr in mp.get_norm_preds():
                tl = print_data(pr, test_list)
            test_list = display_sd(test_list)
    else:
        print("COMBO")
        for mp in multi_preds:
            test_list = []
            print(Task.get(mp.task_id).description, "COMBO")
            for pr in mp.get_all_preds():
                tl = print_data(pr, test_list)
            test_list = display_sd(test_list)

import pdb;pdb.set_trace()
# show_shap_stats([mpa, rpa])
# show_shap_stats([rc, bc], combo=True)



# Standard deviation of list 
# Using sum() + list comprehension 
# mean = sum(test_list) / len(test_list) 
# variance = sum([((x - mean) ** 2) for x in test_list]) / len(test_list) 
# res = variance ** 0.5
 

import pdb;pdb.set_trace()

# rpa.show_norm_scores(first_model_features=False, use_cat=True, include_accuracy=True, axis=True)
bc.show_predictor_scores(['RandomForest', 'CatBoost'], reverse_order=False, non_norm=False, abs_val=False, use_cat=True, include_accuracy=True, axis=False)
# bc.show_predictor_scores(['RandomForest', 'CatBoost'], reverse_order=False, non_norm=False, abs_val=False, use_cat=True, include_accuracy=True, axis=True)
rc.show_predictor_scores(['RandomForest', 'XGBoost'], reverse_order=False, non_norm=False, abs_val=False, use_cat=True, include_accuracy=True, axis=False)
# rc.show_predictor_scores(['RandomForest', 'XGBoost'], reverse_order=True, non_norm=False, abs_val=False, use_cat=True, include_accuracy=True, axis=True)

import pdb;pdb.set_trace()

# mpa.show_predictor_scores(['CatBoost', 'XGBoost'], reverse_order=True, non_norm=False, abs_val=True, use_cat=True, include_accuracy=True, axis=False)
# rpa.show_norm_scores(['RandomForest', 'CatBoost', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)
# mpa.show_abs_scores(['RandomForest', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)
# rpa.show_abs_scores(['RandomForest', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)


def save_all_shap_vals(mpa, rpa, bc, rc):
    bdp = mpa.get_preds()
    rdp = rpa.get_preds()

    mpa.save_shap()
    mpa.save_abs_shap()
    mpa.save_norm_shap()

    rpa.save_shap()
    rpa.save_abs_shap()
    rpa.save_norm_shap()

    bc.save_combo_shap()
    rc.save_combo_shap()

    bap = mpa.get_abs_preds()
    rap = rpa.get_abs_preds()

    mpa.save_shap_values(preds=bap)
    rpa.save_shap_values(preds=rap)

    bnp = mpa.get_norm_preds()
    rnp = rpa.get_norm_preds()

    mpa.save_shap_values(preds=bnp)
    rpa.save_shap_values(preds=rnp)

    bc.save_shap_values(non_norm=False, title="combo")
    rc.save_shap_values(non_norm=False, title="combo")
    print("DONE!")

# save_all_shap_vals(mpa, rpa, bc, rc)

# def save_cp_shap(mp


MatrixPlotter.show(bc.get_all_acc(), rc.get_all_acc(), h_one="Block Combo", h_two="Ring Combo")


import pdb;pdb.set_trace()

for pr in reversed(mpa.get_norm_preds()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)
print("ABS")
for pr in reversed(mpa.get_abs_preds()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)
print("done")
for pr in reversed(mpa.get_predictors()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)

print("multi")
for pr in bc.get_all_preds():
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)

print("ring!!")
for pr in reversed(rpa.get_norm_preds()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)
print("ABS")
for pr in reversed(rpa.get_abs_preds()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)
print("done")
for pr in reversed(rpa.get_predictors()):
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)

print("ring multi")
for pr in rc.get_all_preds():
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, pr.sensor_id, pr.skip_boruta)



bc.show_predictor_scores(['RandomForest'], reverse_order=True, non_norm=False)


import pdb;pdb.set_trace()

# import pdb;pdb.set_trace()
# mpa.gen_train_combo_mp(use_norm_pred=True, get_sg_count=True)

# for pr in reversed(mpa.get_norm_predictors()):
#     if pr.sensor_id == 8 or pr.sensor_id == 7 or pr.sensor_id == 31:
#         print("skip")
#     else:
#         print(Sensor.get(pr.sensor_id).name)
#         pr.train_from(use_shap=True, get_sg_count=True)

# for pr in reversed(mpa.get_abs_predictors()):
#     print(Sensor.get(pr.sensor_id).name)
#     pr.train_from(use_shap=True, get_sg_count=True)

# for pr in reversed(mpa.get_predictors()):
#     print(Sensor.get(pr.sensor_id).name)
#     pr.train_from(use_shap=True, get_sg_count=True)





# print("done with blocks!")

# # for pr in rpa.get_norm_predictors():

# #     pr.train_from(use_shap=True, get_sg_count=True)

# for pr in rpa.get_abs_predictors():
#     pr.train_from(use_shap=True, get_sg_count=True)

# for pr in rpa.get_predictors():
#     pr.train_from(use_shap=True, get_sg_count=True)

# rpa.gen_train_combo_mp(use_norm_pred=True, get_sg_count=True)

# print("done with rings!")
# import pdb;pdb.set_trace()

# print(len(bc.get_all_preds()))
# print(len(rc.get_all_preds()))
# import pdb;pdb.set_trace()
# bc.save_shap_values(abs_val=False, non_norm=False)


# # for rcc in rc.get_all_preds():
# #     rcc.retrain_from(use_shap=True)


# print("Starting Block")
# mpa.gen_train_combo_mp(use_norm_pred=True)

# print("Done with Block!!")

# print("Starting Ring!")
# rpa.gen_train_combo_mp(use_norm_pred=True)

# print("Done with Ring!!")
# import pdb;pdb.set_trace()















# combo ring tasks



# mpa.save_shap_values(abs_val=False, non_norm=False)







print("Done!!")


# for pr in rpb.get_all_preds():
#     pr.train_from(use_shap=True)


# for pr in mpc.get_all_preds():
#     pr.train_from(use_shap=True)


# mpc.save_shap_values(abs_val=False, non_norm=False)
# rpb.save_shap_values(abs_val=False, non_norm=False)



# rpa.gen_train_combo_mp(use_norm_pred=True)







# for nrp in n_preds:
#     nrp.train_from(use_shap=True)


norm_pred =  mpa.get_norm_predictors()

# mpa.gen_scores_for_sensor(non_norm=True, abs_val=True)

mpa.gen_train_combo_mp(use_norm_pred=True, norm_pred=norm_pred)


# bad_abs = mpa.get_abs_predictors()
# good_abs = mpb.get_abs_predictors()

# for ba in bad_abs:
#     ba.delete()

# for ga in good_abs:
#     ga.update(cohort_id=1, multi_predictor_id=mpa.id)


# mpa.gen_scores_for_sensor(non_norm=False, abs_val=True)

# ara = mpcp.get_acc()
# arb = mpcp.get_acc(abs_val=True)
# arc = mpcp.get_acc(non_norm=False)


# ara = mpb.get_acc(abs_val=True)
# arb = mpcp.get_acc(abs_val=True)

# MatrixPlotter.view_and_save_results(ara, results_two=arb, task="Reg")
# MatrixPlotter.view_and_save_results(ara, results_two=arb, task="Reg")



# MatrixPlotter.view_and_save_results(rb, results_two=ra, task="Reg")



# abs value results


results = []
results_reg = []
# Iterate through the mpss list
# for mp in mpss:
#     # Extracting the classifier accuracies
#     accuracies = mp.get_accuracies()['classifier_accuracies']
    
#     # Sorting by accuracy in descending order and rounding to three decimal places
#     sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}

#     print(sorted_accuracies)
    
#     # Append sensor name and its sorted accuracies to the results list
#     results.append((mp.sensor().name, sorted_accuracies))

# # Sort results based on the maximum accuracy value from the sorted accuracies
# results.sort(key=lambda x: max(x[1].values()), reverse=True)

# Iterate through the mpss list

# mps = mpcp.get_predictors()

# for mp in mps:
#     # Extracting the classifier accuracies
#     accuracies = mp.get_accuracies()['classifier_accuracies']
    
#     # Sorting by accuracy in descending order and rounding to three decimal places
#     sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}

#     print(sorted_accuracies)
    
#     # Append sensor name and its sorted accuracies to the results list
#     results_reg.append((mp.sensor().name, sorted_accuracies))

# # Sort results based on the maximum accuracy value from the sorted accuracies
# results_reg.sort(key=lambda x: max(x[1].values()), reverse=True)

# MatrixPlotter.view_and_save_results(results_reg, results_two=results, task="Reg vs ABS")



# Define the tasks, sensor, and classifier names you are interested in
task_descriptions = ['Block_dominant']
sensor_name = 'rsho_x'
classifier_names = ['RandomForest', 'ExtraTrees', 'DecisionTree']  # Your TOP_MODELS list



mpd = MultiPredictor.where(task_id=3, cohort_id=1)


# 30 predictors here (abs, non abs, norm)
default_trials = mpd[0]

# one predictor here (abs combo)
combo_preds = mpd[1]

# for pr in combo_preds.get_predictors():
#     pr.train_from(use_shap=True)


mpcp = MultiPredictor.where(task_id=3, cohort_id=2)


cp_preds = mpcp[0]
MatrixPlotter.view_and_save_results(default_trials.get_acc(), results_two=cp_preds.get_acc(), task="Reg")


mpb.show_predictor_scores(['RandomForest'], reverse_order=True, non_norm=False)



# mpb.show_predictor_scores(['RandomForest'], reverse_order=True, abs_val=True)



# cp
mpcp.show_predictor_scores(['RandomForest'], reverse_order=True, abs_val=True, non_norm=True)
# Gather the PredictorScores
scores_by_task = gather_predictor_scores_by_task(task_descriptions, sensor_name, classifier_names)

# Assuming you want to combine Task 1 and Task 2
if 'Task 1 Description' in scores_by_task and 'Task 2 Description' in scores_by_task:
    for classifier_name in classifier_names:
        # Filter scores by classifier name
        task1_scores = [score for score in scores_by_task['Block_dominant'] if score.classifier_name == classifier_name]
        task2_scores = [score for score in scores_by_task['Rings_dominant'] if score.classifier_name == classifier_name]
        print(task1_scores)
        print(task2_scores)
        # Combine and plot if there are scores for both tasks for the classifier
        if task1_scores and task2_scores:
            # Use the first score for each task (assuming only one score per task-classifier combo)
            combined_plot_filename = combine_shap_beeswarm(task1_scores[0], task2_scores[0], title=f"{classifier_name} {sensor_name} Combined Tasks")
            print(f"Combined SHAP beeswarm plot saved as {combined_plot_filename}")




for sensor_name, accuracies in results:
    print(sensor_name, accuracies)


