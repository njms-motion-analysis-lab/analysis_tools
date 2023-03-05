
import pandas as pd
import csv
import os
import re

class CSVGenerator:
    @classmethod
    def generate_csv(cls, dict1, dict2, filename):
        # Get object names from the first dictionary
        object_names = list(dict1.keys())

        # Create header row with mean and std_dev columns for each object
        header_row = ["pt name"]
        for obj_name in object_names:
            header_row.append(obj_name +  "_mean")
            header_row.append(obj_name +  "_stdev")

        data_rows = []
        for pt_name, pt in dict2.items():
            data_row = [pt_name]
            for header_val in header_row[1:]:
                header_val = header_val.split("_")

                motion_type = header_val[:-1]
                measurement_type = header_val[-1:]
                
                if len(motion_type) != 1:
                    motion_type = "_".join(motion_type)
                else:
                    motion_type = "".join(motion_type)
                
                if motion_type in pt.exp_motions_hash:
                    sample = pt.exp_motions_hash[motion_type]
                    if measurement_type[0] == "mean":
                        data_row.append(sample.mean)
                    elif measurement_type[0] == "stdev":
                        data_row.append(sample.stdev)
                else:
                    data_row.append("")
            data_rows.append(data_row)

        # Write to CSV file
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_row)
            writer.writerows(data_rows)

    @classmethod
    def camel_to_snake(cls, camelcase_string):
        snakecase_string = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', camelcase_string)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', snakecase_string).lower()

    @classmethod
    def combine_csv_files(cls, csv_file1, csv_file2, csv_file3, output_csv):
        # Read in the CSV files
        df1 = pd.read_csv(csv_file1)
        df2 = pd.read_csv(csv_file2)
        df3 = pd.read_csv(csv_file3)

        
        # Append the last character of each CSV filename to the header names
        df1.columns = [col + "_" + csv_file1.split('.')[0][-1] for col in df1.columns]
        df2.columns = [col + "_" + csv_file2.split('.')[0][-1] for col in df2.columns]
        df3.columns = [col + "_" + csv_file3.split('.')[0][-1] for col in df3.columns]

        # Combine the dataframes
        combined_df = pd.concat([df1, df2, df3], axis=1)

        # Write the combined dataframe to a CSV file
        combined_df.to_csv(output_csv, index=False)

    # Searches for CSV files containing the provided substring in the title and combines them.
    @classmethod
    def combine_csvs_with_substring(cls, substring):
        csv_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.csv') and substring in f]
        csv_files = sorted(csv_files)
        if len(csv_files) < 3:
            print(f"Not enough CSV files found containing '{substring}'")
            return None
        else:
            return cls.combine_csv_files(csv_files[0], csv_files[1], csv_files[2], substring + '.csv')