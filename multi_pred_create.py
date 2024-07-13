
# import stuff we need (task)
# create instance of multi predictor
# run multi predictor

# importing...

# import task
from models.legacy_task import Task
from models.legacy_cohort import Cohort
from prediction_tools.legacy_multi_predictor import MultiPredictor


task = Task.where(description="Rings_dominant")[0]
cohort = Cohort.where(name="healthy_controls")[0]


mp = MultiPredictor.find_or_create(
    task_id = task.id,
    cohort_id = cohort.id
)

mp.gen_scores_for_sensor()





