from datetime import datetime
import json
from transitions import Machine

from models.base_model_sqlite3 import BaseModel
from models.legacy_sensor import Sensor

from prediction_tools.time_predictor import TimePredictor
from prediction_tools.legacy_multi_predictor import MultiPredictor


class MultiTimePredictor(BaseModel):
    table_name = "multi_time_predictor"

    # Example states if you want a small workflow:
    states = ["uninitialized", "synchronizing", "ready", "error", "complete"]

    def __init__(self, id=None, task_id=None, codes_score=None, model=None, items=None, created_at=None, updated_at=None, cohort_id=None, multi_predictor_id=None, multi_predictor_feature_id=None):
        self.id = id
        self.task_id = task_id
        self.codes_score = codes_score
        self.model = model
        self.items = items
        self.cohort_id = cohort_id
        self.created_at = created_at
        self.updated_at = updated_at
        self.multi_predictor_id = multi_predictor_id
        self.multi_predictor_feature_id = multi_predictor_feature_id
        # Initialize transitions (optional)
        self.machine = Machine(model=self, states=MultiTimePredictor.states, initial="uninitialized")
        self._init_state_machine()

    def get_multi_predictor(self):
        return MultiPredictor.get(self.multi_predictor_feature_id)
    
    def get_predictors(self):
        return self.get_multi_predictor()

    
    def get_time_predictors(self):
        return TimePredictor.where(multi_time_predictor_id=self.id)
    

    # a roundabout way of getting time predictors that ensures the association works...
    def get_norm_time_predictors(self):
        norm = []
        tps = self.get_time_predictors()
        mp = MultiPredictor.get(self.multi_predictor_feature_id)
        nps = mp.get_norm_preds()
        
        for tp in tps:
            for np in nps:
                if tp.predictor_feature_id == np.id:
                    norm.append(tp)
        return norm
    
    gntp = get_norm_time_predictors

    def get_r2(self):
        return self.get_results("r2")
    
    def get_mse(self):
        return self.get_results("mse")
    
    def get_mae(self):
        return self.get_results("mae")
    
    def get_bcvmse(self):
        return self.get_results("best_cv_mse")
    
    def get_all_results(self):
        return self.get_results(False)
    
    def get_results(self, attr="r2"):
        ntps = self.gntp()
        results = []
        for ntp in ntps:
            snr_results = {}
            snr = Sensor.get(ntp.sensor_id).name
            for ps in ntp.gps():
                cr = json.loads(ps.continuous_results)
                if attr is not False:
                    snr_results[ps.classifier_name] = cr[attr]
                else:
                    snr_results[ps.classifier_name] = cr
            
            results.append((snr, snr_results))

        return results



    def get_abs_predictors(self):
        norm = []
        tps = self.get_time_predictors()
        mp = MultiPredictor.get(self.multi_predictor_feature_id)
        nps = mp.get_norm_preds()
        
        for tp in tps:
            for np in nps:
                if tp.predictor_feature_id == np.id:
                    norm.append(tp)
        return norm





    def _init_state_machine(self):
        """Defines transitions for MultiTimePredictor's AASM-like workflow."""
        self.machine.add_transition(
            trigger="start_sync",
            source="uninitialized",
            dest="synchronizing",
            after="on_start_sync",
        )
        self.machine.add_transition(
            trigger="sync_success",
            source="synchronizing",
            dest="ready",
            after="on_sync_success",
        )
        self.machine.add_transition(
            trigger="sync_failure",
            source="synchronizing",
            dest="error",
            after="on_sync_failure",
        )
        self.machine.add_transition(
            trigger="finish",
            source=["ready", "error"],
            dest="complete",
            after="on_finish",
        )

    # ---------------------------
    # State Machine Callbacks
    # ---------------------------
    def on_start_sync(self):
        print("[MultiTimePredictor FSM] Starting synchronization logic...")

    def on_sync_success(self):
        print("[MultiTimePredictor FSM] Sync completed successfully! Now in 'ready' state.")

    def on_sync_failure(self):
        print("[MultiTimePredictor FSM] An error occurred during sync. Transitioned to 'error' state.")

    def on_finish(self):
        print("[MultiTimePredictor FSM] Done. Moved to 'complete' state.")


    # ---------------------------
    # CRUD / DB Methods
    # ---------------------------
    def save(self):
        """
        Save the MultiTimePredictor record to the database. 
        Leveraging the BaseModel.update(...) or create(...) method.
        """
        # If self.id is None, it doesn't exist in the DB yet; call create().
        if self.id is None:
            created_id = self.create(
                task_id=self.task_id,
                codes_score=self.codes_score,
                model=self.model,
                items=self.items,
                cohort_id=self.cohort_id,
                multi_predictor_id=self.multi_predictor_id,
                multi_predictor_feature_id=self.multi_predictor_feature_id,
            )
            self.id = created_id
        else:
            # Otherwise, update the record
            self.update(
                task_id=self.task_id,
                codes_score=self.codes_score,
                model=self.model,
                items=self.items,
                cohort_id=self.cohort_id,
                multi_predictor_id=self.multi_predictor_id,
                multi_predictor_feature_id=self.multi_predictor_feature_id,
            )
        return self.id

    def create_from_multi_predictors(self, feature_mp=None, timing_mp=None):
        """
        Initializes this MultiTimePredictor by referencing the two given 
        MultiPredictor objects: one for features, one for time.
        """
        # Example: copy relevant fields from the feature multi-predictor\
        self.update(
            task_id = feature_mp.task_id,
            model = feature_mp.model,
            cohort_id = feature_mp.cohort_id,
            multi_predictor_feature_id = feature_mp.id,
        )
        
        
        # If timing_mp is provided, store that link
        if timing_mp:
            self.update(multi_predictor_id = timing_mp.id)

        print(f"[create_from_multi_predictors] MultiTimePredictor saved. ID={self.id}.")
        return self

    def combine_features_and_time(self, feature_mp=None, timing_mp=None):
        """
        Given a feature-based MultiPredictor (feature_mp) and a time-based MultiPredictor (timing_mp),
        iterate through all feature predictors, find the matching time predictor(s),
        and create/associate a new TimePredictor instance pointing to this MultiTimePredictor.
        The match is based on (task_id, sensor_id, cohort_id) and, if multiple match,
        we narrow further by non_norm, then abs_val.
        """
        # 1. If we haven't started sync, do so:
        if self.state == "uninitialized":
            self.start_sync()

        try:
            feature_predictors = feature_mp.get_all_preds()  # list of classification Predictors
            timing_mp_predictors = timing_mp.get_all_preds()     # list of time-based Predictors

            for feat_pred in feature_predictors:
                # 2. Narrow down to time predictors with same (task_id, sensor_id, cohort_id)
                if feature_mp != timing_mp:
                    matching_times = [
                        tpred for tpred in timing_mp_predictors
                        if (tpred.task_id == feat_pred.task_id and
                            tpred.sensor_id == feat_pred.sensor_id and
                            tpred.cohort_id == feat_pred.cohort_id)
                    ]

                    # 3. If more than one, match on non_norm
                    if len(matching_times) > 1 and hasattr(feat_pred, 'non_norm'):
                        matching_times = [
                            tpred for tpred in matching_times
                            if getattr(tpred, 'non_norm', None) == feat_pred.non_norm
                        ]

                    # 4. If still more than one, match on abs_val


                    if not matching_times:
                        # No exact match found; decide if you want to skip or create a default TimePredictor.
                        print(f"No matching time predictor found for feature predictor {feat_pred.id}. Skipping.")
                        continue

                    # 5. If one or multiple remain after filtering, pick the first
                    matched_time_pred = matching_times[0]
                    print(f"Matched time predictor {matched_time_pred.id} for feature predictor {feat_pred.id}.")
                else:
                    print(f"Matched time predictor already matches feature predictor {feat_pred.id}.")
                    matched_time_pred = feat_pred

                # 6. Check if a TimePredictor row linking *this MultiTimePredictor* and *feat_pred* already exists
                existing_timepredictors = TimePredictor.where(
                    predictor_feature_id=feat_pred.id,
                    multi_time_predictor_id=self.id
                )
                if existing_timepredictors:
                    # Already created one; just update its predictor_id if needed
                    existing_tp = existing_timepredictors[0]
                    existing_tp.update(predictor_id = matched_time_pred.id)
                    print(f"TimePredictor {existing_tp.id} already existed, updated predictor_id={matched_time_pred.id}.")
                else:
                    # 7. Create a new TimePredictor row that references this MultiTimePredictor
                    new_time_pred = TimePredictor.find_or_create(
                        predictor_feature_id=feat_pred.id,
                        multi_time_predictor_id=self.id
                    )
                    # Set fields from feature predictor, plus link to matched time predictor
                    new_time_pred.update(
                        task_id = feat_pred.task_id,
                        sensor_id = feat_pred.sensor_id,
                        cohort_id = feat_pred.cohort_id,
                        predictor_id = matched_time_pred.id,
                    )
                    
                    # If your TimePredictor also needs non_norm/abs_val set, do that here:
                    # new_time_pred.non_norm = feat_pred.non_norm
                    # new_time_pred.abs_val = feat_pred.abs_val
                    # new_time_pred.save()

                    print(f"Created TimePredictor {new_time_pred.id} for feature predictor {feat_pred.id}, "
                        f"linked to time predictor {matched_time_pred.id}.")

            # 8. If we get here, presumably the sync is successful
            self.sync_success()

        except Exception as e:
            print(f"Error in combine_features_and_time: {e}")
            self.sync_failure()

        # 9. Mark final state
        self.finish()
        return True