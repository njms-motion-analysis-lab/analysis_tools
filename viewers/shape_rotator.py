import re
from models.sensor import Sensor
from models.task import Task
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from models.gradient_set import GradientSet


class ShapeRotator:
    # Given a gradient set, display the sets subgradients (and corresponding ones) on one chart.
    # Imo this is a good way to sanity check validity and zero value crossing logic.
    def plot_3d_sg(set_instance):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        sensor_id = set_instance.sensor_id
        # Determine the table name based on the class of the instance
        table = re.sub(r'(?<!^)(?=[A-Z])', '_', set_instance.__class__.__name__).lower()
        print(1)
        # find all sensor_ids with the same part, side, and placement
        sensor = Sensor.find_by('id', sensor_id)
        same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)

        sensor_ids = [sensor.id for sensor in same_sensor_ids]
        cls = type(set_instance)
        # get instances for each sensor_id
        set_instances = cls.where(sensor_id=sensor_ids, trial_id=set_instance.trial_id)
        
        # get matrix data from each set_instance
        coords = [instance.mat() for instance in set_instances]
        
        for sub in set_instance.sub_gradients():
            start_time = sub.start_time
            stop_time = sub.stop_time
            sub_coords = []
            for axis_cord in coords:
                sub_coords.append(axis_cord.loc[start_time:stop_time])
            
            ax.plot(sub_coords[0], sub_coords[1], sub_coords[2])
            

        plt.show()


    def plot_3d(set_instance):
        sensor_id = set_instance.sensor_id
        # Determine the table name based on the class of the instance
        table = re.sub(r'(?<!^)(?=[A-Z])', '_', set_instance.__class__.__name__).lower()
        print(1)
        # find all sensor_ids with the same part, side, and placement
        sensor = Sensor.find_by('id', sensor_id)
        same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)

        sensor_ids = [sensor.id for sensor in same_sensor_ids]
        cls = type(set_instance)
        # get instances for each sensor_id
        set_instances = cls.where(sensor_id=sensor_ids, trial_id=set_instance.trial_id)
        
        # get matrix data from each set_instance
        coords = [instance.mat() for instance in set_instances]

        # plot in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(coords[0], coords[1], coords[2])
        plt.show()


    def plot_3ds(set_instances):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

        for idx, set_instance in enumerate(set_instances):
            # get sensor_id from set_instance
            sensor_id = set_instance.sensor_id
            
            # get table name
            table = re.sub(r'(?<!^)(?=[A-Z])', '_', set_instance.__class__.__name__).lower()
            
            # find all sensor_ids with the same part, side, and placement
            sensor = Sensor.find_by('id', sensor_id)
            same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)

            sensor_ids = [sensor.id for sensor in same_sensor_ids]
            trial_ids = [instance.trial_id for instance in set_instances]
            trial_ids = list(set(trial_ids))
            # get class of set_instance
            cls = type(set_instance)

            # get instances for each sensor_id
            
            grouped_instances = {}
            for trial_id in trial_ids:
                grouped_instances[trial_id] = cls.where(trial_id=trial_id, sensor_id=sensor_ids)
            # Plot the grouped instances
            
            for trial_id, instances in grouped_instances.items():
                # get matrix data from each instance
                coords = [instance.mat() for instance in instances]
                # plot in 3D with a different color for each trial_id
                ax.plot(coords[0], coords[1], coords[2])
        
        
        
        plt.show()





tsk = Task.where(id=3)[0]
print(tsk.description)
snr = Sensor.where(name='rwra_x')[0]
tnr = Sensor.where(name='rwrb_x')[0]
gss = tsk.get_gradient_sets_for_sensor(snr)
pss = tsk.get_position_sets_for_sensor(snr)
gs = GradientSet.all()[3]
ps = gs.get_position_set()
sid = gs.sensor_id
mini = pss[:4]

# ShapeRotator.plot_3d(pss[0])
# ShapeRotator.plot_3d(gss[0])
ShapeRotator.plot_3d_sg(gss[0])