from multiprocessing import freeze_support
import os
import sqlite3
from creation_services.old_generator import OldGenerator
from models.legacy_patient import Patient
from models.legacy_patient_task import PatientTask
from models.legacy_sensor import Sensor
from models.legacy_cohort import Cohort
from migrations.legacy_table import Table
import pdb


RAW_DATA_FOLDER = "raw_data/CP_filteredandtrimmed_2024.07.21/Block"

if __name__ == '__main__':
    freeze_support()
    exp = {}
    root_dir = "controls_alignedCoordinateSystem"
    Table.create_tables()
    Table.update_tables()
    # Table.create_and_set_cohort()

    def fix_pd_et_cohort_ids():
        g1 = ['amg__S008','amg__S009','amg__S011','amg__S015','amg__S016','amg__S019','amg__S022','amg__S027','amg__S028','amg__S029','amg__S030','amg__S031',]
        c1 = 'group_1_analysis_me'
        pc1 = Cohort.where(name=c1)[0]
        pg1 = Patient.where(name=g1)
        for pt in pg1:
            pt.update(cohort_id=pc1.id)
            ptts = PatientTask.where(patient_id=pt.id)
            for ptt in ptts:
                ptt.update(cohort_id=pc1.id)
            

        g2 = ['amg__S010','amg__S013','amg__S014','amg__S017','amg__S018','amg__S021','amg__S023','amg__S024','amg__S025','amg__S026',]
        c2 = 'group_2_analysis_me'
        pc2 = Cohort.where(name=c2)[0]
        pg2 = Patient.where(name=g2)
        for pt in pg2:
            pt.update(cohort_id=pc2.id)
            ptts = PatientTask.where(patient_id=pt.id)
            for ptt in ptts:
                ptt.update(cohort_id=pc2.id)
            
        g3 = ['amg__S003','amg__S007','amg__S012','amg__S020',]
        c3 = 'group_3_analysis_me'
        pc3 = Cohort.where(name=c3)[0]
        pg3 = Patient.where(name=g3)
        for pt in pg3:
            pt.update(cohort_id=pc3.id)
            ptts = PatientTask.where(patient_id=pt.id)
            for ptt in ptts:
                ptt.update(cohort_id=pc3.id)
        

    def fix_cp_cohort_ids():
        g1 = ['S008_cp', 'S002_cp', 'S001_cp', 'S006_cp', 'S003_cp', 'S012_cp', 'S004_cp', 'S018_cp']
        c1 = 'cp_before'
        pc1 = Cohort.where(name=c1)[0]
        pg1 = Patient.where(name=g1)
        for pt in pg1:
            pt.update(cohort_id=pc1.id)
            ptts = PatientTask.where(patient_id=pt.id)
            for ptt in ptts:
                print(pc1.id)
                ptt.update(cohort_id=pc1.id)



    def create_sensor_from_string(sensor_string):
        side_map = {"l": "left", "r": "right"}
        side = side_map.get(sensor_string[0], None)
        # if sensor_string is 'lelbm_x':
        #     import pdb
        #     pdb.set_trace()
        part = sensor_string[1:3]
        
        placement = sensor_string[3]
        if len(sensor_string) > 6:
            placement = sensor_string[3:5]
        axis = sensor_string[-1]
        Sensor.find_or_create(name=sensor_string, side=side, axis=axis, placement=placement, part=part)

    def generate_sensors():
        # sensors = []
        sensors = [
            "lwra_x",
            "lwrb_x",
            "lwra_y",
            "lwrb_y",
            "lwra_z",
            "lwrb_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "rfrm_x",
            "rfrm_y",
            "rfrm_z",
            "lelb_x",
            "lelbm_x",
            "lelb_y",
            "lelbm_y",
            "lelb_z",
            "lelbm_z",
            "relb_x",
            "relbm_x",
            "relb_y",
            "relbm_y",
            "relb_z",
            "relbm_z",
            "lupa_x",
            "lupa_y",
            "lupa_z",
            "rupa_x",
            "rupa_y",
            "rupa_z",
            "lsho_x",
            "lsho_y",
            "lsho_z",
            "rsho_x",
            "rsho_y",
            "rsho_z",
            "lwra_x",
            "lwrb_x",
            "lwra_y",
            "lwrb_y",
            "lwra_z",
            "lwrb_z",
            "rwra_x",
            "rwrb_x",
            "rwra_y",
            "rwrb_y",
            "rwra_z",
            "rwrb_z",
            "lfrm_x",
            "lfrm_y",
            "lfrm_z",
            'lfhd_x',
            'lfhd_y',
            'lfhd_z',
            'rfhd_x', 
            'rfhd_y', 
            'rfhd_z', 
            'lbhd_x',
            'lbhd_y', 
            'lbhd_z', 
            'rbhd_x', 
            'rbhd_y', 
            'rbhd_z',
            'rfin_x', 
            'rfin_y', 
            'rfin_z',
            'lfin_x', 
            'lfin_y', 
            'lfin_z',
        ]
        for s_string in sensors:
            create_sensor_from_string(s_string)

    generate_sensors()
    fix_cp_cohort_ids()
    fix_pd_et_cohort_ids()

    files = []
    print("yo")
    for subdir, _, filenames in os.walk(RAW_DATA_FOLDER):
        for filename in filenames:
            file_path = os.path.join(subdir, filename)
            if file_path.lower().endswith('.npy'):
                files.append(file_path)

    n = 0
    import pdb;pdb.set_trace()
    for file in files:
        print(file.lower())
        # Modify this `if` statement to select the files of choice.
        # since we are using the cp file, go to the subdirectory with finished or close to finished trials (i.e. skip s009)
        if "cp" in file.lower():
            cohort = Cohort.find_or_create(name="cp_before", is_control=False, is_treated=False)
        else:
            cohort = Cohort.find_or_create(name="heathy_controls", is_control=True, is_treated=False)
        print(cohort.name)
        if "dynamic" not in file.lower():
            print(file)
            gen = OldGenerator(file, cohort)
            gen.generate_models()
            
            n += 1
            print(file,"finished", n, "features complete")
        else:
            print("skipping", file)
    else:
        print("nope", file.lower())
        

print("Done!")

