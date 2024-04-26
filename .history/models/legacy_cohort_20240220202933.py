from datetime import datetime
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient_task import PatientTask
from models.legacy_trial import Trial

class Cohort(LegacyBaseModel):
    table_name = "cohort"
    
    def __init__(self, id=None, name=None, is_control=None, is_treated=None, description=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.is_control = is_control
        self.is_treated = is_treated
        self.description = description
    

    def get_patient_tasks(self):
        return PatientTask.where(cohort_id=self.id)
    
    def get_alt_cohort(self):
        hc_cohort = Cohort.find_by("name", HC_COHORT)
        pd_cohort = Cohort.find_by("name", PD_COHORT)
        et_cohort = Cohort.find_by("name", ET_COHORT)

    
    def get_trials(self):
        pts = self.get_patient_tasks()
        trials = []
        for pt in pts:
            trials.concat(Trial.where(patient_task_id=pt.id))
        return trials



