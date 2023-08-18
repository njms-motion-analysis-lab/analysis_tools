# pr = MultiPredictor.all()[0].get_predictors()[0]# import sqlite3
# from models.base_model_sqlite3 import BaseModel as LegacyBaseModel

# from migrations.legacy_table import Table as LegacyTable
import csv
import pdb
import re
import pandas as pd
from models.legacy_task import Task

# from models.legacy_gradient_set import GradientSet
# from models.legacy_sensor import Sensor
import os

from models.legacy_patient_task import PatientTask
from models.legacy_patient import Patient
from models.legacy_gradient_set import GradientSet
import numpy as np
from models.legacy_sub_gradient import SubGradient
from collections import defaultdict
from models.legacy_sensor import Sensor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
# from viewers.shape_rotator import ShapeRotator
from viewers.matrix_plotter import MatrixPlotter
import seaborn as sns
import matplotlib.pyplot as plt
from prediction_tools.legacy_multi_predictor import MultiPredictor
snr = Sensor.all()
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

all_patients = Patient.all()
# for patient in all_patients:
#     patient.delete_duplicate_trials()

from models.legacy_trial import Trial

# def replace_axis_labels(directory):
#     for filename in os.listdir(directory):
#         if filename.endswith(".png"):
#             filepath = os.path.join(directory, filename)
            
#             # Read the current title
#             with open(filepath, "rb") as file:
#                 content = file.read()
#                 title_start = content.find(b"tEXtTitle")
#                 title_end = content.find(b"\x00", title_start)
#                 title = content[title_start+9:title_end].decode("latin-1")
            
#             # Replace the axis portion of the title
#             new_title = title.replace("x", "x (anterior-posterior)").replace("y", "y (medial-lateral)").replace("z", "z (superior-inferior)")
            
#             # Modify the title in the PNG file
#             modified_content = content[:title_start+9] + new_title.encode("latin-1") + content[title_end:]
            
#             # Save the modified PNG file
#             with open(filepath, "wb") as file:
#                 file.write(modified_content)



# dir = 'parallel_plots/grad_data__kurtosis'
# replace_axis_labels(dir)
# for t in tasks:
#     t.gen_all_stats_csv(abs_val=False, non_normed=True)


from prediction_tools.legacy_predictor import Predictor
from models.legacy_task import Task
from models.legacy_sensor import Sensor


# LegacyTable.update_tables()
from models.legacy_task import Task

t = Task.all()[2]
# Retrieve all instances of the Trial class
# all_trials = Task.all()

# Iterate through each trial instance
# for task in all_trials:
#     if "Dominant" in task.description and "Nondominant" not in task.description:
#         task.update(is_dominant=True)
#     if "Nondominant" in task.description or "nondominant" in task.description:
#         task.update(is_dominant=False)
#     if "dominant" in task.description and "nondominant" not in task.description:
#         task.update(is_dominant=True)

#     for trial in task.trials():
#         trial.update(is_dominant=task.is_dominant)

print("done")



snr = Sensor.all()





SENSOR_CODES = [
    'rbhd_x',
    'relb_x',
    'relbm_x',
    'rfhd_x',
    'rfin_x',
    'rfrm_x',
    'rsho_x',
    'rupa_x',
    'rwra_x',
    'rwrb_x',
]

from constants import ALLOWED_SENSORS


tsk = Task.dominant()
dom_tsks = []

for tk in tsk:
    if 'Ring'.lower() in tk.description.lower() or 'Block'.lower() in tk.description.lower():
        dom_tsks.append(tk)
 

# for tkk in dom_tsks:
#     for sn in Sensor.where(name=ALLOWED_SENSORS):
#         print(sn.name, tkk.description, len(tkk.get_gradient_sets_for_sensor(sn)))



snr = Sensor.where(name=SENSOR_CODES)
dt = Task.dominant()



dtt = dt[0]
# print("task", dtt.description)
snr = Sensor.all()[7]
# print("sensor:", snr.name)



sens = Sensor.all()
sss = Sensor.where(axis='x', side='right')
# for sen in sss:
#     print(sen.name)
mps = []

new_accuracies = []

mps = Predictor.where(multi_predictor_id=1)


ring_results = []
# Iterate through the mpss list
for mp in mps:
    # Extracting the classifier accuracies
    accuracies = mp.get_accuracies()['classifier_accuracies']
    
    # Sorting by accuracy in descending order and rounding to three decimal places
    sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}
    
    # Append sensor name and its sorted accuracies to the ring_results list
    ring_results.append((mp.sensor().name, sorted_accuracies))

# Sort ring_results based on the maximum accuracy value from the sorted accuracies
ring_results.sort(key=lambda x: max(x[1].values()), reverse=True)

# Create a list to store sensor names and their sorted accuracies
results = []
mpss = Predictor.where(multi_predictor_id=2)
import pdb;pdb.set_trace()
print("Blocks")
for mp in mpss:
    mp.retrain_from()
    print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_accuracies'], mp.get_accuracies()['classifier_metrics'])

print("Rings")
for mp in mps:
    mp.retrain_from()
    print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_accuracies'], mp.get_accuracies()['classifier_metrics'])







# mpss[9].retrain_from()
# MatrixPlotter.plot_matrices(mps, 'LogisticRegression')
# MatrixPlotter.plot_matrices(mpss, 'LogisticRegression')

# MatrixPlotter.plot_matrices(mps, 'RandomForest')
# MatrixPlotter.plot_matrices(mps, 'DecisionTree')
# MatrixPlotter.plot_matrices(mps, 'ExtraTrees')


# MatrixPlotter.plot_matrices(mpss, 'RandomForest')
# MatrixPlotter.plot_matrices(mpss, 'DecisionTree')
# MatrixPlotter.plot_matrices(mpss, 'ExtraTrees')



# Iterate through the mpss list
for mp in mpss:
    # Extracting the classifier accuracies
    accuracies = mp.get_accuracies()['classifier_accuracies']
    
    # Sorting by accuracy in descending order and rounding to three decimal places
    sorted_accuracies = {k: round(v, 3) for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}
    
    # Append sensor name and its sorted accuracies to the results list
    results.append((mp.sensor().name, sorted_accuracies))

# Sort results based on the maximum accuracy value from the sorted accuracies
results.sort(key=lambda x: max(x[1].values()), reverse=True)
# Print the sorted results
for sensor_name, accuracies in results:
    print(sensor_name, accuracies)

print("block results")
for sensor_name, accuracies in results:
    print(sensor_name, accuracies)

# Print the sorted ring_results
print("ring results")
for sensor_name, accuracies in ring_results:
    print(sensor_name, accuracies)








import pdb;pdb.set_trace()


print("Rings")
for mp in mps:
    print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_accuracies'], mp.get_accuracies()['classifier_metrics'])
    print()
    

print("Blocks")
for mp in mpss:
    print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_accuracies'], mp.get_accuracies()['classifier_metrics'])
    print()






import pdb;pdb.set_trace()












# def view_and_save_results(results, results_two=None, task=None):
#     # Convert the list to a dictionary
#     results_dict = {sensor: accuracies for sensor, accuracies in results}
#     if results_two is not None:
#         results_two_dict = {sensor: accuracies for sensor, accuracies in results_two}

#     # Get all model names
#     all_models = set()
#     for accuracies in results_dict.values():
#         all_models.update(accuracies.keys())
#     all_models = sorted(list(all_models))

#     # Remove unwanted models
#     unwanted_models = ['LightGBM', 'NaiveBayes']
#     for model in unwanted_models:
#         try:
#             all_models.remove(model)
#         except ValueError:
#             pass  # Model not in the list, so simply pass

#     # Write to CSV
#     with open('output.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(['Sensor'] + all_models)  # Writing header
        
#         for sensor, accuracies in results_dict.items():
#             row = [sensor.replace('_x', '')]
#             for model in all_models:
#                 row.append(accuracies.get(model, 'N/A'))  # Appending model accuracy, if not found write 'N/A'
#             writer.writerow(row)

#     # Convert the results into a dataframe for easy plotting
#     df = pd.DataFrame(columns=['Sensor'] + all_models)

#     for sensor, accuracies in results_dict.items():
#         row = {'Sensor': sensor.replace('_x', '')}
#         for model in all_models:
#             row[model] = accuracies.get(model, None)
#         df = df.append(row, ignore_index=True)

#     # Order columns by their highest score
#     ordered_columns = ['Sensor'] + df.drop('Sensor', axis=1).max().sort_values(ascending=False).index.tolist()
#     if results_two is not None:
#         for sensor, accuracies in results_two_dict.items():
#             row = {'Sensor': sensor.replace('_x', '')}
#             for model in all_models:
#                 row[model] = accuracies.get(model, None)
#             df = df.append(row, ignore_index=True)
    

#     df = df[ordered_columns]

#     # Heatmap
#     plt.figure(figsize=(12, 8))
#     sns.heatmap(df.set_index('Sensor').astype(float), cmap="YlGnBu", annot=True, cbar_kws={'label': 'Accuracy'}, yticklabels=1)
#     if task is not None and results_two is None:
#         plt.title(f"Model Accuracies Across Sensors for {task} Task")  # Using provided method to get title
#     else:
#         plt.title(f"Model Accuracies Across Sensors")  # Using provided method to get title

#     plt.xticks(rotation=45)
#     plt.yticks(rotation=0)  # This ensures that sensor names are horizontal
#     plt.show()

# MatrixPlotter.view_and_save_results(results, ring_results)

# import pdb;pdb.set_trace()

# view_and_save_results(ring_results, "Ring")

# import pdb;pdb.set_trace()

# import pdb;pdb.set_trace()

# mps[0].retrain_from()
# print("Rings")
# for mp in mps:
    
#     print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_metrics'])
#     print()
    # print(mp.get_accuracies()['classifier_params'])
    # print()

# print()
# print()
# print("Blocks")
# for mp in mpss:
#     print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_metrics'])
#     print()

# import pdb;pdb.set_trace()


print()
print()
print("Block Tasks...")


print("Ring Tasks...")
# for mp in mps:
#     mp.retrain_from()
#     print(Sensor.get(mp.sensor_id).name, mp.get_accuracies()['classifier_metrics'])
#     print()





    



import pdb;pdb.set_trace()










# print("starting...")


mp = MultiPredictor.find_or_create(task_id=dom_tsks[1].id)
mp.gen_scores_for_sensor()
mps.append(MultiPredictor(task=dt))


print("done!...")
import pdb; pdb.set_trace()



# print()
# # non normalized
# print("Normalized...")



# Empty DataFrame to store best features
best_features_df = pd.DataFrame()

# Initialize a dictionary to store data
best_rows = defaultdict(list)

# Iterate over each index

accuracies = {}
bdf = pd.DataFrame()

for col in items[0].df.columns:
    if col in ['is_dominant', 'patient']:
        continue
    accuracies[col] = 0
    cdf = pd.Series()
    for item in items:
        idf = item.df[col]
        iacc = item.accuracies.get(col, 0) # It's good practice to use the get() method to avoid KeyErrors
        if iacc > accuracies[col]:
            accuracies[col] = iacc
            cdf = idf
    bdf[col] = cdf

# Add 'is_dominant' and 'patient' columns from the first item
bdf['is_dominant'] = items[0].df['is_dominant']
bdf['patient'] = items[0].df['patient']



# for index in range(len(items[0].df)):
#     # For each index, find the item (predictor) with the highest accuracy
#     best_item = max(items, key=lambda x: x.accuracies[x.features_dom[index]])

#     # Add the row from the best predictor's dataframe to our new dictionary
#     for feature in best_item.features_dom:
#         best_rows[feature].append(best_item.df[feature].iloc[index])

#     # Also keep track of the patient and is_dominant column
#     best_rows['patient'].append(best_item.df['patient'].iloc[index])
#     best_rows['is_dominant'].append(best_item.df['is_dominant'].iloc[index])


# Create a dataframe from our dictionary
best_df = bdf



# Prepare the feature set X with the best features and patient
X = best_df.drop(columns=['is_dominant'])

# Prepare the target variable y
y = best_df['is_dominant']

# Instantiate the RandomForestClassifier
model = RandomForestClassifier()

# Fit the model
model.fit(X, y)

# Perform K-fold cross-validation and store accuracies
cv_scores = cross_val_score(model, X, y, cv=5)

# Print the average accuracy
print('Accuracy:', np.mean(cv_scores))


