from exp_motion_sample_trial import ExpMotionSampleTrial
from models.base_model import BaseModel
import sqlite3

from models.patient_motion import PatientMotion



class Trial(BaseModel):
    table_name = "trial"

    def __init__(self, id=None, patient_motion_id=None, timestamp=None, name=None, matrix=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.patient_motion_id = patient_motion_id
        self.timestamp = timestamp

    # def create(self, **kwargs):
    #     self.patient_motion_id = kwargs.get("patient_motion_id")
    #     self.timestamp = kwargs.get("timestamp")
    #     print(f"Creating Trial with kwargs: {kwargs}")  # Add this print statement
    #     return super().create(**kwargs)

    def update(self, **kwargs):
        self.patient_motion_id = kwargs.get("patient_motion_id", self.patient_motion_id)
        self.timestamp = kwargs.get("timestamp", self.timestamp)
        print(f"Updating Trial with kwargs: {kwargs}")  # Add this print statement
        return super().update(**kwargs)
    
    def generate_sets(self, data):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        PositionSet = import_module("models.position_set").PositionSet
        Sensor = import_module("models.sensor").Sensor
        
        position_data = data['positional']
        gradient_data = data['gradients']
        sensors = Sensor.all()

        kc = 0
        kcl = len(sensors)
        kl = len(position_data.keys()[1:2])
        for key in position_data.keys()[1:2]:
            col = position_data[key]
            cc = 0
            sensor = Sensor.get_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)
            if sensor.id:
                print(sensor.id)
                ps = PositionSet.find_or_create(sensor_id=sensor.id, trial_id=self.id, matrix=col)
                
                
                cc += 1
            else:
                print("noooooo")
            kc += 1
            

        kcl = len(sensors)
        kl = len(gradient_data.keys()[:1])
        for key in gradient_data.keys()[1:2]:
            col = position_data[key]
            cc = 0
            sensor = Sensor.get_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)
            if sensor.id:
                print(sensor.id)
                gs = GradientSet.find_or_create(sensor_id=sensor.id, trial_id=self.id, matrix=col)
                mt = ExpMotionSampleTrial(key, sensor.name, measurements=data)
                import pdb
                pdb.set_trace()
                cc += 1

        print("Done 3")

    def get_patient(self):
        from importlib import import_module
        Patient = import_module("models.motion").Patient
        patient_motion = PatientMotion.get(id=self.patient_motion_id)
        if patient_motion:
            return Patient.get(id=patient_motion.patient_id)
        return None

    def get_motion(self):
        from importlib import import_module
        Motion = import_module("models.motion").Motion
        patient_motion = PatientMotion.get(id=self.patient_motion_id)
        if patient_motion:
            return Motion.get(id=patient_motion.motion_id)
        return None


    def create_sensor_from_string(self, sensor_string):
        from importlib import import_module
        Sensor = import_module("models.sensor").Sensor
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        
        part = sensor_string[1:3]
        iteration = sensor_string[3]
        axis = sensor_string[-1]
        sensor = Sensor.find_or_create(name=sensor_string, side=side, axis=axis, iteration=iteration, part=part)
        return sensor