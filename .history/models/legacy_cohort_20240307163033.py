from datetime import datetime
from models.base_model_sqlite3 import BaseModel as LegacyBaseModel
from models.legacy_patient_task import PatientTask
from models.legacy_trial import Trial

OPPO_MAPPING = {
    "group_1_analysis_me":"group_2_analysis_me",
    "group_2_analysis_me":"group_3_analysis_me",
    "group_3_analysis_me":"group_1_analysis_me",
}
    
# merge PD and ET into one group
# second step is to sep. neuro disorders

class Cohort(LegacyBaseModel):
    table_name = "cohort"

    def __init__(self, id=None, name=None, is_control=None, is_treated=None, description=None, created_at=None, updated_at=None):
        super().__init__()
        self.id = id
        self.name = name
        self.is_control = is_control
        self.is_treated = is_treated
        self.description = description
    
    def is_alt_compare(self):
        return (
            self.name == "group_1_analysis_me" or
            self.name == "group_2_analysis_me" or
            self.name == "group_3_analysis_me"
        )
    
    def get_patient_tasks(self):
        return PatientTask.where(cohort_id=self.id)
    
    def get_alt_cohort(self):
        alt_name = OPPO_MAPPING.get(self.name)
        if alt_name:
            return Cohort.find_by("name", alt_name)
        else:
            return None

    def get_trials(self):
        pts = self.get_patient_tasks()
        trials = []
        for pt in pts:
            trials.concat(Trial.where(patient_task_id=pt.id))
        return trials



