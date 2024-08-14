import pickle
from exp_motion_sample_trial import ExpMotionSampleTrial
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient_task import PatientTask

MAX_GRADIENT_SET_NUM = 60
class Trial(LegacyBaseModel):
    table_name = "trial"
    

    def __init__(self, id=None, name=None, patient_task_id=None, trial_num=None, timestamp=None, matrix=None, created_at=None, updated_at=None, is_dominant=True):
        super().__init__()
        self.id = id
        self.name = name
        self.patient_task_id = patient_task_id
        self.trial_num = trial_num
        self.timestamp = timestamp
        self.is_dominant = is_dominant
    
    def get_gradient_sets(self):
        from importlib import import_module
        GradientSet = import_module("models.legacy_gradient_set").GradientSet
        return GradientSet.where(trial_id=self.id)

    def update(self, **kwargs):
        self.patient_task_id = kwargs.get("patient_task_id", self.patient_task_id)
        self.timestamp = kwargs.get("timestamp", self.timestamp)
        
        return super().update(**kwargs)
    
    def generate_sets(self, data, skip_pos=False, force_update=False):
        from importlib import import_module
        GradientSet = import_module("models.legacy_gradient_set").GradientSet
        PositionSet = import_module("models.legacy_position_set").PositionSet
        Sensor = import_module("models.legacy_sensor").Sensor
        gss = GradientSet.where(trial_id=self.id)
        
        if len(gss) == MAX_GRADIENT_SET_NUM:
            print("already finished name:", self.name, "id:", self.id, "pt_id", self.patient_task_id)
            return
        
        gradient_data = data['gradients']

        def update_existing_position_set(sensor_id, key, col):
            print("UPDATING POSITION")
            try:
                ps = PositionSet.where(sensor_id=sensor.id, name=key, trial_id=self.id)[0]
                if ps.mat().equals(col) == False:
                    col_serialized = pickle.dumps(col)
                    ps = ps.update(matrix=col_serialized)
                else:
                    print("SAME POSITION SET")
                return ps
            except Exception as e:
                print("COULD NOT UPDATE EXISTING POSITION SET", e)
                return None

        # Remove array slicing to include all sensors
        if not skip_pos:
            position_data = data['positional']
            for key in position_data.keys():
                col = position_data[key]
                print(key)
                key = self.extract_sensor_name(key)
                
                sensor = Sensor.find_by('name', key)
                if not sensor:
                    print("next...")
                    continue
               
                existing_position_set = None

                if force_update==False:
                    try:
                        PositionSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)
                    except ValueError:
                        print("position set already exists for trial", self.id, self.name)
                        return
                else:
                    existing_position_set = update_existing_position_set(sensor, key, col)

    
        def update_existing_gradient_set(sensor, key, col):
            print("UPDATING GRADIENT")
            try:
                gs = GradientSet.where(sensor_id=sensor.id, name=key, trial_id=self.id)[0]
                if gs.mat().equals(col) == False:
                    col_serialized = pickle.dumps(col)
                    gs.update(matrix=col_serialized)
                else:
                    print("SAME GS")
                    return gs, True
                
                return gs, False
            except:
                print("COULD NOT UPDATE EXISTING GRADIENT SET")
                return None, False
    
        # Remove array slicing to include all sensors
        for key in gradient_data.keys():
            col = gradient_data[key]
            print(key)
            key = self.extract_sensor_name(key)
            sensor = Sensor.find_by('name', key)
            if not sensor:
                print("next...")
                continue
            else:
                if not existing_position_set and len(PositionSet.where(sensor_id=sensor.id, name=key, trial_id=self.id)) == 0:
                    PositionSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)


                if force_update==False:
                    print(f"Generating data for sensor {sensor.name}!!!")
                    grad_set = GradientSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)
                else:
                    print(f"Updating data for sensor {sensor.name}!!!")
                    grad_set, same_gs = update_existing_gradient_set(sensor, key, col)
                    if same_gs is True:
                        print("SKIPPING Creation due to same GS")
                        continue
                    else:
                        if grad_set is None:
                            print(f"GRAD SET NOT FOUND! Generating data for sensor {sensor.name}!!!")
                            grad_set = GradientSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)
        
                grad_set.create_subgradients()
                grad_set.update(aggregated_stats=grad_set.calc_aggregate_stats())
                print(f"Done with {sensor.name}!!!")
            
    
    # TODO: [Stephen] combine the dynamic and, uh, static models/methods/logic.
    def extract_sensor_name(self, header):
        """
        Extracts the sensor name from a given header string.

        Parameters:
        - header (str): The header string from which to extract the sensor name.

        Returns:
        - str: The extracted sensor name.
        """
        # Remove the 'gradient_' prefix
        sensor_name_with_unit = header.replace("velocity_", "")
        
        # Now, remove the unit part, assuming it is always enclosed in parentheses
        # This will split the string into parts using '(' as delimiter and take the first part
        sensor_name = sensor_name_with_unit.split('(')[0].strip()
        
        return sensor_name

    def patient(self):
        from importlib import import_module
        Patient = import_module("models.legacy_patient").Patient
        patient_task = PatientTask.get(id=self.patient_task_id)
        if patient_task:
            return Patient.get(id=patient_task.patient_id)
        return None

    def task(self):
        from importlib import import_module
        Task = import_module("models.legacy_task").Task
        patient_task = PatientTask.where(id=self.patient_task_id)[0]
        if patient_task:
            return Task.get(id=patient_task.task_id)
        return None
    
    @classmethod
    def select(cls, trials, **kwargs):
        def match(trial):
            return all(getattr(trial, key) == value for key, value in kwargs.items())
        
        return [trial for trial in trials if match(trial)]


    def create_sensor_from_string(self, sensor_string):
        from importlib import import_module
        Sensor = import_module("models.legacy_sensor").Sensor
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        
        part = sensor_string[1:3]
        placement = sensor_string[3]
        axis = sensor_string[-1]
        sensor = Sensor.find_or_create(name=sensor_string, side=side, axis=axis, placement=placement, part=part)
        new_sensor = Sensor.find_by("name", sensor_string)
        if new_sensor.id:
            return new_sensor
        else:
            raise NameError
