from asyncio import Task
from models.legacy_gradient_set import GradientSet
from models.legacy_patient import Patient
from models.legacy_patient_task import PatientTask
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from models.legacy_trial import Trial
import os


class Progress:
    def by_patient(print_path=False):
        tsk = Task.all()
        mega = []
        for tk in tsk:
            print(tk.description, tk.id)
            ppt = PatientTask.where(task_id=tk.id)
            print("    ",len(ppt))
            extra = []
            for pp in ppt:
                tr = Trial.where(patient_task_id=pp.id)
                extra.append(len(tr))
                
            mega.append(sum(extra))
            print("          ", extra, sum(extra))
        print(sum(mega))
        pts = Patient.all()

        for pt in pts:
            pt_tasks = PatientTask.where(patient_id=pt.id)
            if pt_tasks:
                seen = {}
                pt_trial_count = len(pt.trials())
                print(pt.name, "num tasks:", len(pt_tasks), ", num trials:", pt_trial_count)
                trials = PatientTask.sort_by(pt.trials(),'name')
                for trial in trials:

                    gss = GradientSet.where(trial_id=trial.id)
                    if len(gss) != 0:
                        counts = 0
                        for gs in gss:
                            if gs.aggregated_stats != None:
                                counts += 1
                        
                        agg_stats = counts

                    else:
                        agg_stats = 0
                    
                    # if len(dss) != 0:
                    #     d_counts = 0
                    #     for ds in dss:
                    #         if ds.aggregated_stats != None:
                    #             d_counts += 1
                        
                    #     d_agg_stats = d_counts

                    # else:
                    d_agg_stats = 0
                    # if trial.name in seen:
                    #     st = seen[trial.name]
                    #     zss = GradientSet.where(trial_id=st.id)
                        # print("           name:", st.name,   "trial_id:", st.id,  "num gradients:", len(zss))
                        # st.delete_with_kids()
                        # if len(gss) == 0:
                        #     trial.delete_with_kids()
                    print("         ", trial.name,"id:", trial.id,   "trial_pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.updated_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats)
                    
                            

                    

            #         if agg_stats != len(gss):
            #             print('          ^ error ^', "gradient_set id:", gss[0].id)
            # print("      ")
        
        root_dir = "controls_alignedCoordinateSystem"

        if print_path is True:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    print(file_path)

Progress.by_patient()
import pdb;pdb.set_trace()
print("yo")