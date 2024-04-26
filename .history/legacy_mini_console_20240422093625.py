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
import numpy as np
from models.legacy_sub_gradient import SubGradient
from collections import defaultdict
from models.legacy_sensor import Sensor
from sklearn.model_selection import cross_val_score, GridSearchCV
from models.legacy_cohort import Cohort
from sklearn.ensemble import RandomForestClassifier
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
from models.legacy_trial import Trial
from prediction_tools.legacy_multi_predictor import MultiPredictor

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
        et_cohort,
        pd_cohort,
        hc_cohort,
    ]

    mpc = []

    def fix_mistake(mp):
        pra = Predictor.where(task_id=mp.task_id, cohort_id=hc_cohort.id)
        print("fixing", len(pra), "task a")
        for pa in pra:
            pa.update(multi_predictor_id=mp.id)
        
        print("done")

        
    
    for balance_task in balance_tasks:
        # print("Creating/Finding MP for:", balance_task.description, cohort.name)
        mp = MultiPredictor.find_or_create(
            task_id = balance_task.id,
            cohort_id = hc_cohort.id,
        )
        import pdb;pdb.set_trace()
        mp.sensors = sensors
        # mp.gen_scores_for_sensor(skip_default_sensors=True, force_load=True, add_other=True)
        mpc.append(
            mp
        )
    
    for mp in mpc:
        mp.save_shap()
        mp.save_abs_shap()
        mp.save_norm_shap()
    import pdb;pdb.set_trace()
    
    # print(len(mpc), "MP to gen found")
    for mp in reversed(mpc):
        desc = Task.get(mp.task_id).description
        sensors = mp.sensors
        print("Generating Scores for: ", desc, "Sensor: ", sensors, "PRED LEN", len(mp.get_all_preds()))
        
    pdb.set_trace()
    print("done")
import pdb;pdb.set_trace()
bc = MultiPredictor.where(model="norm_non_abs_combo")[0]
# mpc.aggregate_shap_values(non_norm=False)
# feature_cluster_map = mpc.feature_cluster_map(non_norm=False)
# bc.show_norm_scores(axis=True)

    
# fetch_new_tasks()

pr = Predictor.where(multi_predictor_id=6)
# healthy controls bloc
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

# MatrixPlotter.show(mpa.get_acc(), rpa.get_acc(), alt=True, h_one="Block", h_two="Ring")

# MatrixPlotter.show(mpa.get_abs_acc(), rpa.get_abs_acc(), alt=True, h_one="Block Absolute Values", h_two="Ring Absolute Values")

# MatrixPlotter.show(mpa.get_norm_acc(), rpa.get_norm_acc(), alt=True, h_one="Block Normalized", h_two="Ring Normalized")
# mpa.get_all_preds()[-1].train_from(get_sg_count=True)
bc = MultiPredictor.get(7)
rc = MultiPredictor.get(8)
test_list = []

for pr in bc.get_norm_preds():
    num_features = len(pr.get_predictor_scores()[0].get_top_n_features(500))
    test_list.append(num_features)
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, Sensor.get(pr.sensor_id).name, "num features:", num_features)

# printing list 
print("The original list : " + str(test_list)) 
 
# Standard deviation of list 
# Using sum() + list comprehension 


test_list = []
for pr in rc.get_norm_preds():
    num_features = len(pr.get_predictor_scores()[0].get_top_n_features(500))
    test_list.append(num_features)
    print(pr.id, pr.updated_at, pr.non_norm, pr.abs_val, Sensor.get(pr.sensor_id).name, "num features:", num_features)


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

show_shap_stats([mpa, rpa])
show_shap_stats([rc, bc], combo=True)

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

mpa.show_predictor_scores(['CatBoost', 'XGBoost'], reverse_order=True, non_norm=False, abs_val=True, use_cat=True, include_accuracy=True, axis=False)
rpa.show_norm_scores(['RandomForest', 'CatBoost', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)
mpa.show_abs_scores(['RandomForest', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)
rpa.show_abs_scores(['RandomForest', 'XGBoost'], reverse_order=True, use_cat=True, include_accuracy=True, axis=False)


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



MatrixPlotter.show(bc.get_all_acc(), rc.get_all_acc(), alt=True, h_one="Block Combo", h_two="Ring Combo")


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





print("done with blocks!")

# for pr in rpa.get_norm_predictors():

#     pr.train_from(use_shap=True, get_sg_count=True)

for pr in rpa.get_abs_predictors():
    pr.train_from(use_shap=True, get_sg_count=True)

for pr in rpa.get_predictors():
    pr.train_from(use_shap=True, get_sg_count=True)

rpa.gen_train_combo_mp(use_norm_pred=True, get_sg_count=True)

print("done with rings!")
import pdb;pdb.set_trace()

print(len(bc.get_all_preds()))
print(len(rc.get_all_preds()))
import pdb;pdb.set_trace()
bc.save_shap_values(abs_val=False, non_norm=False)


# for rcc in rc.get_all_preds():
#     rcc.retrain_from(use_shap=True)


print("Starting Block")
mpa.gen_train_combo_mp(use_norm_pred=True)

print("Done with Block!!")

print("Starting Ring!")
rpa.gen_train_combo_mp(use_norm_pred=True)

print("Done with Ring!!")
import pdb;pdb.set_trace()

















# combo ring tasks



# mpa.save_shap_values(abs_val=False, non_norm=False)







print("Done!!")
import pdb;pdb.set_trace()

# for pr in rpb.get_all_preds():
#     pr.train_from(use_shap=True)


# for pr in mpc.get_all_preds():
#     pr.train_from(use_shap=True)


mpc.save_shap_values(abs_val=False, non_norm=False)
rpb.save_shap_values(abs_val=False, non_norm=False)



rpa.gen_train_combo_mp(use_norm_pred=True)



rpa.gen_scores_for_sensor(force_load=True)
rpa.gen_scores_for_sensor(abs_val=True, force_load=True)
rpa.gen_scores_for_sensor(non_norm=False, force_load=True)



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


