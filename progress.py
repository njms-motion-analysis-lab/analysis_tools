from asyncio import Task
from models.gradient_set import GradientSet
from models.dynamic_gradient_set import DynamicGradientSet
from models.patient import Patient
from models.patient_task import PatientTask
from models.sensor import Sensor
from models.task import Task
from models.trial import Trial
import os


class Progress:
    def by_patient(print_path=False):
        pts = Patient.all()

        for pt in pts:
            pt_tasks = PatientTask.where(patient_id=pt.id)
            if pt_tasks:
                pt_trial_count = len(pt.trials())
                print(pt.name, "num tasks:", len(pt_tasks), ", num trials:", pt_trial_count)
                for trial in pt.trials():

                    gss = GradientSet.where(trial_id=trial.id)
                    dss = DynamicGradientSet.where(trial_id=trial.id)
                    if len(gss) is not 0:
                        counts = 0
                        for gs in gss:
                            if gs.aggregated_stats is not None:
                                counts += 1
                        
                        agg_stats = counts

                    else:
                        agg_stats = 0
                    
                    if len(dss) is not 0:
                        d_counts = 0
                        for ds in dss:
                            if ds.aggregated_stats is not None:
                                d_counts += 1
                        
                        d_agg_stats = d_counts

                    else:
                        d_agg_stats = 0

        
                    if agg_stats is not None:
                        print("     name:", trial.name,   "trial_id:", trial.id,  "num gradients:", len(gss), "num aggregate stats:", agg_stats, "num_dynamic_gradients:", len(dss), "num_dynamic_stats:", d_agg_stats)
                    else:
                        print("     name:", trial.name,   "num gradients:", len(gss), "num aggregate stats:", agg_stats)
                    
                    if agg_stats != len(gss):
                        print('          ^ error ^', "gradient_set id:", gss[0].id)
            print("      ")
        
        root_dir = "controls_alignedCoordinateSystem"

        if print_path is True:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    print(file_path)

        import pdb;pdb.set_trace()
Progress.by_patient()