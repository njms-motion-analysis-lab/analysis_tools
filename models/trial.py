from exp_motion_sample_trial import ExpMotionSampleTrial
from models.base_model import BaseModel
from models.patient_task import PatientTask

class Trial(BaseModel):
    table_name = "trial"

    def __init__(self, id=None, name=None, patient_task_id=None, timestamp=None, matrix=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.patient_task_id = patient_task_id
        self.timestamp = timestamp

    def update(self, **kwargs):
        self.patient_task_id = kwargs.get("patient_task_id", self.patient_task_id)
        self.timestamp = kwargs.get("timestamp", self.timestamp)
        
        return super().update(**kwargs)
    
    def generate_sets(self, data):
        from importlib import import_module
        GradientSet = import_module("models.gradient_set").GradientSet
        PositionSet = import_module("models.position_set").PositionSet
        Sensor = import_module("models.sensor").Sensor
        
        position_data = data['positional']
        gradient_data = data['gradients']

        # Remove array slicing to include all sensors
        for key in position_data.keys()[1:2]:
            col = position_data[key]
            sensor = Sensor.find_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)
            PositionSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)
    
        # Remove array slicing to include all sensors
        for key in gradient_data.keys()[1:2]:
            col = gradient_data[key]
            sensor = Sensor.find_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)
            GradientSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)
            grad_set = GradientSet.where(trial_id=self.id)[0]
            grad_set.create_subgradients()
            grad_set.update(aggregated_stats=grad_set.calc_aggregate_stats())
            #print("IM HERE RN:",GradientSet.where(trial_id=self.id)[0].get_aggregate_stats())
    
    # TODO: [Stephen] combine the dynamic and, uh, static models/methods/logic.
    def generate_dynamic_gradient_sets(self, data):
        from importlib import import_module
        DynamicGradientSet = import_module("models.dynamic_gradient_set").DynamicGradientSet
        DynamicPositionSet = import_module("models.dynamic_position_set").DynamicPositionSet

        dynamic_position_data = data['positional']
        dynamic_gradient_data = data['gradients']

        position_data = data['positional']
        Sensor = import_module("models.sensor").Sensor

            # Remove array slicing to include all sensors
        for key in position_data.keys()[1:2]:
            col = dynamic_position_data[key]
            sensor = Sensor.find_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)
            DynamicPositionSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)

        for key in dynamic_gradient_data.keys()[1:2]:
            col = dynamic_gradient_data[key]
            sensor = Sensor.find_by('name', key)
            if not sensor:
                sensor = self.create_sensor_from_string(key)

            dgs = DynamicGradientSet.find_or_create(sensor_id=sensor.id, name=key, trial_id=self.id, matrix=col)

            if dgs.id is None:
                pass
            try:
                dgs.where(trial_id=self.id)[0]
                dgs.create_dynamic_subgradients()
                dgs.update(aggregated_stats=dgs.calc_aggregate_stats())
            except IndexError:
                pass
                



    def patient(self):
        from importlib import import_module
        Patient = import_module("models.task").Patient
        patient_task = PatientTask.get(id=self.patient_task_id)
        if patient_task:
            return Patient.get(id=patient_task.patient_id)
        return None

    def task(self):
        from importlib import import_module
        Task = import_module("models.task").Task
        patient_task = PatientTask.get(id=self.patient_task_id)
        if patient_task:
            return Task.get(id=patient_task.task_id)
        return None


    def create_sensor_from_string(self, sensor_string):
        from importlib import import_module
        Sensor = import_module("models.sensor").Sensor
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
