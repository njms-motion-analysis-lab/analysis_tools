import os
from models.sensor import Sensor
from models.trial import Trial
import numpy as np
from models.task import Task
from models.patient import Patient
from models.base_model import BaseModel
import pdb

class Generator:
    def __init__(self, filename: str, conn=None, cursor=None):
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
        self.data = np.load(filename, allow_pickle=True).tolist()
        self.name = os.path.splitext(os.path.basename(filename))[0]
        self.patient = ""
        self.task = ""
        self.conn = None
        self.cursor = None
        self.dynamic = False
        self.set_attributes()
        self.generate_models()
        
    
    def set_attributes(self):
        # Extract the patient name, experimental task description, and variant from the `name` attribute of the instance.
        # If the variant is not the same as the experimental task description, append it to the experimental task description.
        # If the experimental task description ends with an underscore, remove it.
        # Set the `patient` and `task` attributes of the instance to the extracted values.
        sub_dir = self.name.split('_')
        root, patient, exp_task, variant = sub_dir[0], sub_dir[1], sub_dir[2], sub_dir[-1]
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
        c_patient = Patient.find_or_create(name=self.patient)
        c_task = Task.find_or_create(description=self.task)
        
        c_patient.add_task(c_task)
        pm = c_patient.patient_task_by_task(c_task)
        
        counts = 0
        for key, value in self.data.items():
            tr = Trial.find_or_create(name=key, patient_task_id=pm.id, trial_num=counts)

            if not self.dynamic:
                tr.generate_sets(data=value)
            else:
                tr.generate_dynamic_gradient_sets(data=value)
            counts += 1
        
        print("done")





