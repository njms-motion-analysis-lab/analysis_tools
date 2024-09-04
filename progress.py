from asyncio import Task
import os
from models.legacy_gradient_set import GradientSet
from models.legacy_patient import Patient
from models.legacy_patient_task import PatientTask
from viewers.shape_rotator import ShapeRotator
from models.legacy_sensor import Sensor
from models.legacy_task import Task
from models.legacy_trial import Trial
from viewers.shape_rotator import ShapeRotator


class Progress:
    def by_patient(print_path=False):
        tsk = Task.all()
        mega = []
        for tk in tsk:
            print(tk.description, tk.id)
            ppt = PatientTask.where(task_id=tk.id)
            # print("    ",len(ppt))
            extra = []
            for pp in ppt:
                tr = Trial.where(patient_task_id=pp.id)
                extra.append(len(tr))
                
            # mega.append(sum(extra))
            # print("          ", extra, sum(extra))
        # print(sum(mega))
        pts = Patient.where(cohort_id=2)
        zeros = []
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
                        normalized = 0
                        abs_val = 0
                        ss_counts = 0
                        ss_normalized = 0
                        ss_abs_val = 0
                        for gs in gss:
                            # print(gs.updated_at)
                            # gs.fix_reg_sub_stats(force=True)
                            if gs.aggregated_stats != None:
                                counts += 1
                            if gs.normalized != None:
                                normalized += 1
                            if gs.abs_val != None:
                                abs_val += 1
                            
                            if gs.set_stats_non_norm != None:
                                ss_counts += 1
                            if gs.set_stats_norm != None:
                                ss_normalized += 1
                            if gs.set_stats_abs != None:
                                ss_abs_val += 1
                        agg_stats = counts
                        agg_normalized = normalized
                        agg_abs_val = abs_val

                        agg_ss_stats = ss_counts
                        agg_ss_normalized = ss_normalized
                        agg_ss_abs_val = ss_abs_val
                    else:
                        print("ZERO")
                        agg_stats = 0
                    
                    # if len(dss) != 0:
                    #     d_counts = 0
                    #     for ds in dss:
                    #         if ds.aggregated_stats != None:
                    #             d_counts += 1
                        
                    #     d_agg_stats = d_counts

                    # else:
                    d_agg_stats = 0
                    issue = ""
                    if trial.name in seen:
                        issue = issue + " SEEN "
                        st = seen[trial.name]
                        zss = GradientSet.where(trial_id=st.id)
                        # print(issue, trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.created_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats, "num nomalized:", agg_normalized)
                        # st.delete_with_kids()
                        # if len(gss) == 0:
                        #     trial.delete_with_kids()
                        # trial.delete_self_and_children()
                        extras = []
                    if agg_stats != 0:
                        tk = Task.get(PatientTask.where(id=trial.patient_task_id)[0].task_id)
                        
                        if agg_stats != 60:
                            issue = issue + " STATS NUM "
                            # print("         ", trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.created_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats)

                        if(abs(len(trial.name) - len(tk.description))) > 1:
                            issue = issue + " MISMATCH "
                            # print("      ", abs(len(trial.name) - len(tk.description)))
                            # print("         ", trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.created_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats)
                            # trial.delete_self_and_children()
                        if int(trial.name[-1]) != trial.trial_num + 1:
                            issue = issue + " NUMBER"
                            # print("      ", trial.name[-1].lower(), trial.trial_num)
                            # print("          ", trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.created_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats)    
                            # trial.delete_self_and_children()
                            
                        if tk.id == 1 or tk.id == 2:
                            # print(issue, trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num, "num gradients:", len(gss), "num aggregate stats:", agg_stats,"normalized:", agg_normalized, "abs_val", agg_abs_val, "ss_stats", agg_ss_stats,"ss_normalized:", agg_ss_normalized, "ss_abs_val", agg_ss_abs_val)
                            pass
                        else:
                            print(issue, trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num, "num gradients:", len(gss), "num aggregate stats:", agg_stats,"normalized:", agg_normalized, "abs_val", agg_abs_val, "ss_stats", agg_ss_stats,"ss_normalized:", agg_ss_normalized, "ss_abs_val", agg_ss_abs_val)
                        
                        # if (issue == " STATS NUM  NUMBER"):
                        #     print(issue, trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num, "num gradients:", len(gss), "num aggregate stats:", agg_stats,"normalized:", agg_normalized, "abs_val", agg_abs_val, "ss_stats", agg_ss_stats,"ss_normalized:", agg_ss_normalized, "ss_abs_val", agg_ss_abs_val)
                        #     # trial.delete_self_and_children()
                    else:
                        issue = issue + " zero "
                        # print(issue, trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num, "num gradients:", len(gss), "num aggregate stats:", agg_stats,"normalized:", agg_normalized, "abs_val", agg_abs_val)
                        # trial.delete_self_and_children()
                        zeros.append(trial)
                    seen[trial.name] = trial

            #         if agg_stats != len(gss):
            #             print('          ^ error ^', "gradient_set id:", gss[0].id)
            # print("      ")
        
        root_dir = "controls_alignedCoordinateSystem"
        print("ZZZZ")
        print(seen)
        for trial in zeros:
            print("         ", trial.name,"id:", trial.id,   "pt_id:", trial.patient_task_id, "pt_task_name", tk.description,"pt_task_id", tk.id, "trial_name", trial.task().description, "trial_num:", trial.trial_num,  "created_at", trial.updated_at, "num gradients:", len(gss), "num aggregate stats:", agg_stats)
            # trial.delete_self_and_children()

        if print_path is True:
            for subdir, _, files in os.walk(root_dir):
                for file in files:
                    file_path = os.path.join(subdir, file)
                    print(file_path)

Progress.by_patient()
import pdb;pdb.set_trace()
print("yo")