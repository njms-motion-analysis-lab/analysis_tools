import os
from models.sensor import Sensor
from models.trial import Trial
import numpy as np
from models.motion import Motion
from models.patient import Patient
from models.base_model import BaseModel
import pdb

class Generator:
    def __init__(self, filename: str, conn=None, cursor=None):
        # Load the data from the specified `filename`, convert it to a dictionary, and store it as an instance attribute.
        # The `name` attribute is set to the base name of the file, without the extension.
        # The `patient` and `motion` attributes are initialized to empty strings, and are set in the `set_attributes` method.
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
        self.motion = ""
        self.conn = None
        self.cursor = None
        self.set_attributes()
        self.generate_models()
        
    
    def set_attributes(self):
        # Extract the patient name, experimental motion description, and variant from the `name` attribute of the instance.
        # If the variant is not the same as the experimental motion description, append it to the experimental motion description.
        # If the experimental motion description ends with an underscore, remove it.
        # Set the `patient` and `motion` attributes of the instance to the extracted values.
        sub_dir = self.name.split('_')
        root, patient, exp_motion, variant = sub_dir[0], sub_dir[1], sub_dir[2], sub_dir[-1]

        if variant != exp_motion:
            exp_motion = exp_motion + '_' + variant

        if exp_motion.endswith('_'):
            exp_motion = exp_motion[:-1]
        
        self.motion = exp_motion
        self.patient = patient
    

    def generate_models(self):
        # Find or create a `Patient` object with the `name` attribute equal to the `patient` attribute of the instance.
        # Find or create a `Motion` object with the `description` attribute equal to the `motion` attribute of the instance.
        # Add the `Motion` object to the list of motions associated with the `Patient` object.
        c_patient = Patient.find_or_create(name=self.patient)
        c_motion = Motion.find_or_create(description=self.motion)
        
        c_patient.add_motion(c_motion)
        pm = c_patient.get_patient_motion_by_motion(c_motion)
        
        counts = 0
        for key, value in self.data.items():
            print(f"key: {key}, pm: {pm}, pm.id: {pm.id}")
            tr = Trial.find_or_create(name=key, patient_motion_id=pm.id, trial_num=counts)
            tr.generate_sets(data=value)
            counts += 1
        
        print("done")





