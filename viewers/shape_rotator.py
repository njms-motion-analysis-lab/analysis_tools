from importlib import import_module
import re
from models.legacy_patient import Patient
from models.legacy_patient_task import PatientTask
from models.legacy_position_set import PositionSet
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from models.legacy_gradient_set import GradientSet
from models.legacy_trial import Trial


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


    def plot_3d(set_instances, title=None):
        if not isinstance(set_instances, list):
            set_instances = [set_instances]

        # Plot in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        for set_instance in set_instances:
            sensor_id = set_instance.sensor_id
            # Determine the table name based on the class of the instance
            table = re.sub(r'(?<!^)(?=[A-Z])', '_', set_instance.__class__.__name__).lower()
            
            # Find all sensor_ids with the same part, side, and placement
            sensor = Sensor.find_by('id', sensor_id)
            same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)

            sensor_ids = [sensor.id for sensor in same_sensor_ids]
            cls = type(set_instance)
            # Get instances for each sensor_id
            sensor_set_instances = cls.where(sensor_id=sensor_ids, trial_id=set_instance.trial_id)
            
            # Get matrix data from each set_instance
            coords = [instance.mat() for instance in sensor_set_instances]
            
            # Plot coordinates
            ax.plot(coords[0], coords[1], coords[2], label=f'{sensor.side} Hand')
        
        # Labels and legend
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        if title is not None:
            ax.set_title(title)
        ax.legend()
        
        plt.show()
        
    def get_other_set(set_instance):
        sensor_id = set_instance.sensor_id
        
        # Determine the table name based on the class of the instance
        table = re.sub(r'(?<!^)(?=[A-Z])', '_', set_instance.__class__.__name__).lower()
        sensor = Sensor.find_by('id', sensor_id)
        opposite_id = Task.get_counterpart_sensor(sensor.name)
        opposite_sensor = Sensor.find_by('id', opposite_id)

        same_sensor_ids = Sensor.where(part=sensor.part, side=sensor.side, placement=sensor.placement)
        opposite_sensor_ids = Sensor.where(part=opposite_sensor.part, side=opposite_sensor.side, placement=opposite_sensor.placement)

        sensor_ids = [sensor.id for sensor in same_sensor_ids]
        oppopsite_ids = [opp_sensor.id for opp_sensor in opposite_sensor_ids]

        cls = type(set_instance)




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

    def plot_by_pt(patient):
        trials = patient.trials()
        try:
            selected_trial = Trial.select(trials, name="BlockDominant01")[0]
        except IndexError:
            # in case we had a bad trial
            selected_trial = Trial.select(trials, name="BlockDominant02")[0]

        # Fetch sensors for right and left hand coordinates
        sensor_right = Sensor.where(name="rwra_x")[0]
        sensor_left = Sensor.where(name="lwra_x")[0]
        
        # Fetch position sets for the selected trial and both sensors
        position_set_right = PositionSet.where(trial_id=selected_trial.id, sensor_id=sensor_right.id)[0]
        position_set_left = PositionSet.where(trial_id=selected_trial.id, sensor_id=sensor_left.id)[0]
        
        # Plot both right and left hand coordinates
        ShapeRotator.plot_3d([position_set_right, position_set_left], title=patient.name)


# num = 1

# lnr = Sensor.where(name='lwra_x')[0]
# snr = Sensor.where(name='rwra_x')[0]
# max = 0
# min = 10000
# sum = 0
# while num < 26:
#     amt = PatientTask.where(patient_id=num, task_id=4)[0].combined_sub_gradient_stats_list(snr, loc='grad_data__sum_values')['mean']
#     print(amt, num)
    
#     if amt > max:
#         curr = num
#         max = amt
#     if amt < min:
#         curr_m = num
#         min = amt
#     num += 1
#     sum += amt

# print(curr)
# print(curr_m)
# print(sum)
# print("next...")
# max = 0
# min = 10000
# num = 1
# sum = 0
# while num < 26:
#     amt = PatientTask.where(patient_id=num, task_id=4)[0].combined_sub_gradient_stats_list(lnr, loc='grad_data__sum_values')['mean']
#     print(amt, num)
    
#     if amt > max:
#         curr = num
#         max = amt
#     if amt < min:
#         curr_m = num
#         min = amt
#     num += 1
#     sum += amt


# print(curr)
# print(curr_m)
# print(sum)
# print("next...")

# max = 0
# min = 10000
# sum = 0
# num = 1
# # while num < 26:
# #     amt = PatientTask.where(patient_id=num, task_id=3)[0].combined_sub_gradient_stats_list(snr, loc='grad_data__sum_values')['mean']
# #     print(amt, num)
    
# #     if amt > max:
# #         curr = num
# #         max = amt
# #     if amt < min:
# #         curr_m = num
# #         min = amt
# #     num += 1
# #     sum += amt

# # print(curr)
# # print(curr_m)
# # print(sum)
# # print("next...")
# # max = 0
# # min = 10000
# # num = 1
# # sum = 0
# # while num < 26:
# #     amt = PatientTask.where(patient_id=num, task_id=3)[0].combined_sub_gradient_stats_list(snr, loc='grad_data__sum_values')['mean']
# #     print(amt, num)
    
# #     if amt > max:
# #         curr = num
# #         max = amt
# #     if amt < min:
# #         curr_m = num
# #         min = amt
# #     num += 1
# #     sum += amt


# # print(curr)
# # print(curr_m)
# # print(sum)
# # print("next...")








# tsk = Task.where(id=3)[0]
# tsknd = Task.where(id=4)[0]


# tskk = Task.where(id=2)[0]
# tskknd = Task.where(id=1)[0]


# print(tsk.description)
# print(tskk.description)
# snr = Sensor.where(name='rwra_x')[0]
# tnr = Sensor.where(name='rwrb_x')[0]
# gss = tsk.get_gradient_sets_for_sensor(snr)
# pss = tsk.get_position_sets_for_sensor(snr)

# gsss = tskk.get_gradient_sets_for_sensor(snr)
# psss = tskk.get_position_sets_for_sensor(snr)

# ptn = Patient.where(name='S005')[0]
# pti = Patient.where(id=5)[0]
# ptc = Patient.where(id=1)[0]

# pti_ = Patient.where(id=1)[0]
# pti_1 = Patient.where(id=2)[0]
# pti_2 = Patient.where(id=3)[0]
# pti_3 = Patient.where(id=4)[0]
# pti_4 = Patient.where(id=5)[0]
# pti_5 = Patient.where(id=6)[0]
# pti_6 = Patient.where(id=7)[0]
# pti_7 = Patient.where(id=8)[0]
# pti_8 = Patient.where(id=9)[0]
# pti_9 = Patient.where(id=10)[0]
# pti_10 = Patient.where(id=11)[0]
# pti_11 = Patient.where(id=12)[0]
# pti_12 = Patient.where(id=13)[0]
# pti_13 = Patient.where(id=14)[0]
# pti_14 = Patient.where(id=15)[0]
# pti_15 = Patient.where(id=16)[0]
# pti_16 = Patient.where(id=17)[0]
# pti_17 = Patient.where(id=18)[0]
# pti_18 = Patient.where(id=19)[0]
# pti_19 = Patient.where(id=20)[0]
# pti_20 = Patient.where(id=21)[0]
# pti_21 = Patient.where(id=22)[0]
# pti_22 = Patient.where(id=23)[0]
# pti_23 = Patient.where(id=24)[0]
# pti_24 = Patient.where(id=25)[0]
# lnr = Sensor.where(name='lwra_x')[0]






# import pdb;pdb.set_trace()

# tskknd = Task.where(id=1)[0]
# x = pti_.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# x = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]

# x = pti_.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_1.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_2.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_3.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_4.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_5.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_6.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_7.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_8.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_9.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_10.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_11.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_12.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_13.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_14.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_15.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_16.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# # x = pti_17.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_18.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_19.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_20.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0] # this one is the lefty
# x = pti_21.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_22.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_23.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# x = pti_24.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]


# ShapeRotator.plot_3d_sg(x)

# import pdb;pdb.set_trace()
# tpti_ = PatientTask.where(patient_id=1, task_id=4)[0]
# PatientTask.where(patient_id=1, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_1 = PatientTask.where(patient_id=2, task_id=4)[0]
# PatientTask.where(patient_id=2, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_2 = PatientTask.where(patient_id=3, task_id=4)[0]
# PatientTask.where(patient_id=3, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_3 = PatientTask.where(patient_id=4, task_id=4)[0]
# PatientTask.where(patient_id=4, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_4 = PatientTask.where(patient_id=5, task_id=4)[0]
# PatientTask.where(patient_id=5, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_5 = PatientTask.where(patient_id=6, task_id=4)[0]
# PatientTask.where(patient_id=6, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_6 = PatientTask.where(patient_id=7, task_id=4)[0]
# PatientTask.where(patient_id=7, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_7 = PatientTask.where(patient_id=8, task_id=4)[0]
# PatientTask.where(patient_id=8, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_8 = PatientTask.where(patient_id=9, task_id=4)[0]
# PatientTask.where(patient_id=9, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_9 = PatientTask.where(patient_id=10, task_id=4)[0]
# PatientTask.where(patient_id=10, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_10 = PatientTask.where(patient_id=11, task_id=4)[0]
# PatientTask.where(patient_id=11, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_20 = PatientTask.where(patient_id=12, task_id=4)[0]
# PatientTask.where(patient_id=12, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_30 = PatientTask.where(patient_id=13, task_id=4)[0]
# PatientTask.where(patient_id=13, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_40 = PatientTask.where(patient_id=14, task_id=4)[0]
# PatientTask.where(patient_id=14, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_50 = PatientTask.where(patient_id=15, task_id=4)[0]
# PatientTask.where(patient_id=15, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_60 = PatientTask.where(patient_id=16, task_id=4)[0]
# PatientTask.where(patient_id=16, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_70 = PatientTask.where(patient_id=17, task_id=4)[0]
# PatientTask.where(patient_id=17, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_80 = PatientTask.where(patient_id=18, task_id=4)[0]
# PatientTask.where(patient_id=18, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_90 = PatientTask.where(patient_id=19, task_id=4)[0]
# PatientTask.where(patient_id=19, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_50 = PatientTask.where(patient_id=20, task_id=4)[0]
# PatientTask.where(patient_id=20, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_60 = PatientTask.where(patient_id=21, task_id=4)[0]
# PatientTask.where(patient_id=21, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_70 = PatientTask.where(patient_id=22, task_id=4)[0]
# PatientTask.where(patient_id=22, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_80 = PatientTask.where(patient_id=23, task_id=4)[0]
# PatientTask.where(patient_id=23, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_90 = PatientTask.where(patient_id=24, task_id=4)[0]
# PatientTask.where(patient_id=24, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')


# tpti_90 = PatientTask.where(patient_id=25, task_id=4)[0]
# PatientTask.where(patient_id=25, task_id=4)[0].get_gradient_sets_for_sensor(lnr, loc='grad_data__mean')




# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]
# # pti_ = Patient.where(id=)[0]


# bl_gsi = pti.patient_task_by_task(tsk).get_gradient_sets_for_sensor(lnr)[0]


# gsn = ptn.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# gsi = pti.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]





# # block and ring dom
# b_gsi = pti.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# bl_gsi = pti.patient_task_by_task(tsk).get_gradient_sets_for_sensor(lnr)[0]
# bnd_gsi = pti.patient_task_by_task(tsknd).get_gradient_sets_for_sensor(snr)[0]
# bndl_gsi = pti.patient_task_by_task(tsknd).get_gradient_sets_for_sensor(lnr)[0]


# rb_gsi = pti.patient_task_by_task(tskk).get_gradient_sets_for_sensor(snr)[0]
# rbl_gsi = pti.patient_task_by_task(tskk).get_gradient_sets_for_sensor(lnr)[0]
# rbnd_gsi = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# rbndl_gsi = pti.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(lnr)[0]


# # name
# nb_gsi = ptn.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]
# nbl_gsi = ptn.patient_task_by_task(tsk).get_gradient_sets_for_sensor(lnr)[0]
# nbnd_gsi = ptn.patient_task_by_task(tsknd).get_gradient_sets_for_sensor(snr)[0]
# nbndl_gsi = ptn.patient_task_by_task(tsknd).get_gradient_sets_for_sensor(lnr)[0]

# rnb_gsi = ptn.patient_task_by_task(tskk).get_gradient_sets_for_sensor(snr)[0]
# rnbl_gsi = ptn.patient_task_by_task(tskk).get_gradient_sets_for_sensor(lnr)[0]
# rnbnd_gsi = ptn.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(snr)[0]
# rnbndl_gsi = ptn.patient_task_by_task(tskknd).get_gradient_sets_for_sensor(lnr)[0]


# r_gsi = pti.patient_task_by_task(tskk).get_gradient_sets_for_sensor(snr)[0]



# lsi = pti.patient_task_by_task(tskk).get_gradient_sets_for_sensor(lnr)[0]






# gsc = ptc.patient_task_by_task(tsk).get_gradient_sets_for_sensor(snr)[0]


# import pdb;pdb.set_trace()
# # # ShapeRotator.plot_3d(pss[0])
# # # ShapeRotator.plot_3d(gss[0])

# # #  pt id 5, right wrist block => looks normalish
# # ShapeRotator.plot_3d_sg(b_gsi)


# # #  pt id 5, left wrist block => looks weird
# # ShapeRotator.plot_3d_sg(bl_gsi)


# # #  pt id 5, right wrist non dom block
# # ShapeRotator.plot_3d_sg(bnd_gsi)


# # #  pt id 5, left wrist non dom block
# # ShapeRotator.plot_3d_sg(bndl_gsi)





# # #  pt id 5, right wrist ring => looks normalish
# # ShapeRotator.plot_3d_sg(rb_gsi)


# # #  pt id 5, left wrist ring => looks weird
# # ShapeRotator.plot_3d_sg(rbl_gsi)


# # #  pt id 5, right wrist non dom ring
# # ShapeRotator.plot_3d_sg(rbnd_gsi)


# # #  pt id 5, left wrist non dom ring
# # ShapeRotator.plot_3d_sg(rbndl_gsi)







# # #  pt name S005, right wrist block => looks still
# # ShapeRotator.plot_3d_sg(nb_gsi)


# # #  pt name S005, left wrist block => looks still
# # ShapeRotator.plot_3d_sg(nbl_gsi)


# # #  pt name S005, right wrist non dom block
# # ShapeRotator.plot_3d_sg(nbnd_gsi)


# # #  pt name S005, left wrist non dom block
# # ShapeRotator.plot_3d_sg(nbndl_gsi)



# # #  pt name S005, right wrist ring => looks normalish
# # ShapeRotator.plot_3d_sg(rnb_gsi)


# # #  pt name S005, left wrist ring => looks weird
# # ShapeRotator.plot_3d_sg(rnbl_gsi)


# # #  pt name S005, right wrist non dom ring
# # ShapeRotator.plot_3d_sg(rnbnd_gsi)


# # #  pt name S005, left wrist non dom ring
# # ShapeRotator.plot_3d_sg(rnbndl_gsi)







# # ShapeRotator.plot_3d_sg(x)
# # ShapeRotator.plot_3d_sg(gsss[0])

# ShapeRotator.plot_3d_sg(x)