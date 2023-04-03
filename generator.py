import os
import pdb
from models.sensor import Sensor
import numpy as np
from models.motion import Motion
from models.patient import Patient
from models.base_model import BaseModel

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
        self.generate_sensors()
        
    
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

    def generate_sensors(self):
        sensors = [
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
            "lfrm_x",
            "lfrm_y",
            "lfrm_z",
            "rfrm_x",
            "rfrm_y",
            "rfrm_z",
            "lelb_x",
            "lelbm_x",
            "lelb_y",
            "lelbm_y",
            "lelb_z",
            "lelbm_z",
            "relb_x",
            "relbm_x",
            "relb_y",
            "relbm_y",
            "relb_z",
            "relbm_z",
            "lupa_x",
            "lupa_y",
            "lupa_z",
            "rupa_x",
            "rupa_y",
            "rupa_z",
            "lsho_x",
            "lsho_y",
            "lsho_z",
            "rsho_x",
            "rsho_y",
            "rsho_z",
        ]
        for s_string in sensors:
            self.create_sensor_from_string(s_string)

    def create_sensor_from_string(self, sensor_string):
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        
        part = sensor_string[1:3]
        iteration = sensor_string[3]
        axis = sensor_string[-1]
        patient = Patient.find_or_create(name="cheryl")
        print(patient.name)
        sensor = Sensor.find_or_create(name=sensor_string, side=side, axis=axis, iteration=iteration, part=part)
        print(sensor.id)

