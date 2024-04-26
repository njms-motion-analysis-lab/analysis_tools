import os
import re
import threading
from models.legacy_sensor import Sensor
from models.legacy_trial import Trial
import numpy as np
from models.legacy_task import Task
from models.legacy_patient import Patient
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel


class OldGenerator:
    def __init__(self, filename: str, cohort=None, conn=None, cursor=None, skip_list=False):
        # Load the data from the specified `filename`, convert it to a dictionary, and store it as an instance attribute.
        # The `name` attribute is set to the base name of the file, without the extension.
        # The `patient` and `task` attributes are initialized to empty strings, and are set in the `set_attributes` method.
        # Finally, the `set_attributes` and `generate_models` methods are called to process the data.
        #
        # Input:
        # - `filename`: the path to the file containing the data to process.
        #
        # Output:
        # - None.
        if not skip_list:
            self.data = np.load(filename, allow_pickle=True).tolist()
        else:
            self.data = np.load(filename, allow_pickle=True)
        self.filename = filename
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.patient = ""
        self.task = ""
        self.cohort = cohort
        self.conn = None
        self.cursor = None
        self.dynamic = False
        self.set_analysis_me_attributes()
        self.generate_models()
        
    def set_analysis_me_attributes(self):
        print(self.filename)
        parts = self.filename.split('/')
        
        # Extract cohort name, e.g., "analysis_me_group_1"
        # Assuming 'analysis_me' is constant and group information is variable
        group = parts[2].replace(' ', '_')  # Replace spaces with underscores
        
        # Extract patient name (e.g., "amg1_S019")
        # Assuming the patient code is the fourth part of the path and is uppercase
        patient_code = parts[3].upper()
        cohort_code = group[5]  # Assuming group info starts with 'group ' and we need the number after
        self.patient = f"amg{cohort_code}_{patient_code}"
        
        # Correctly extract task and trial from the last part of the path
        # Splitting the last part of the path by '_', expecting 'Task X_Trial Y.npy'
        task_trial_info = parts[-1].split('_')
        task_part = task_trial_info[0]  # 'Task X'
        trial_part = task_trial_info[1]  # 'Trial Y.npy'
        
        # Extract task number and make it title case, assuming it's always in the format 'Task X'
        task_name = parts[4]  # Assuming 'Gait', 'Balance', etc. is the task name
        task_number = re.search(r'\d+', task_part).group()  # Extract the number from 'Task X'
        self.task = f"{task_name.title()}{task_number.zfill(2)}"  # Pad the task number with zeros
        
        # Extract trial number, assuming it's always in the format 'Trial Y.npy'
        self.trial = int(re.search(r'\d+', trial_part).group())
        
        print(f"Extracted - Cohort: {self.cohort.name}, Patient Name: {self.patient}, Task: {self.task}, Trial: {self.trial}")
    
    def set_attributes(self):
        # Extract the patient name, experimental task description, and variant from the `name` attribute of the instance.
        # If the variant is not the same as the experimental task description, append it to the experimental task description.
        # If the experimental task description ends with an underscore, remove it.
        # Set the `patient` and `task` attributes of the instance to the extracted values.
        sub_dir = self.name.split('_')
        print("hello")
        root, patient, exp_task, variant = sub_dir[0], sub_dir[1], sub_dir[2], sub_dir[-1]
        if self.cohort is not None:
            patient = patient + "_cp"
        if root != 'alignedCoordsByStart':
            self.dynamic = True

        if variant != exp_task:
            exp_task = exp_task + '_' + variant

        if exp_task.endswith('_'):
            exp_task = exp_task[:-1]
        
        self.task = exp_task
        self.patient = patient
    

    def generate_models(self):
        # Find or create a `Patient` object with the `name` attribute equal to the `patient` attribute of the instance.
        # Find or create a `Task` object with the `description` attribute equal to the `task` attribute of the instance.
        # Add the `Task` object to the list of tasks associated with the `Patient` object.
        # Generates columns corresponding to x,y, and z values.
        print("YOLO")
        c_patient = Patient.find_or_create(name=self.patient, cohort_id=self.cohort.id)
        c_task = Task.find_or_create(description=self.task)
        
        c_patient.add_task(c_task)
        pm = c_patient.patient_task_by_task(c_task)
        if pm == None:
            print("empty pt")
        else:    
            counts = 0
            import pdb;pdb.set_trace()
            gradient_data = self.calculate_acceleration_gradients(selfdata)

            for key, value in self.data.items():
                print(key, value)
                
                # tr = Trial.find_or_create(name=key, patient_task_id=pm.id, trial_num=counts)

                # if not self.dynamic:
                #     tr.generate_sets(data=value)
                #     counts += 1
    
    def calculate_acceleration_gradients(self.data):
    # Placeholder for storing gradients
        gradient_data = {}
        
        # Assuming 'data' is a structured array, iterate over its dtype.names to find acceleration columns
        for name in data.dtype.names:
            if "Acceleration" in name:
                # Calculate the derivative of the column
                # np.gradient computes the gradient considering uniform spacing
                derivative = np.gradient(data[name], edge_order=2)
                
                # Store the derivative in gradient_data with an appropriate key
                gradient_key = f"gradient_{name}"
                gradient_data[gradient_key] = derivative
                
        # Assuming there's a way to attach 'gradient_data' back into 'data'
        # This step depends on how 'data' is structured in your actual code
        # For demonstration, returning gradient_data
        return gradient_data

        # Example usage
        # Assuming 'data' is your structured array containing the acceleration data
        
        
        print("done")




