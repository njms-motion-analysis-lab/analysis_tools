import pandas as pd
from models.gradient_set import GradientSet
from tsfresh import extract_features
import seaborn as sns

def extract_tsfresh_features(matrix_df):
    matrix_df['id'] = 0
    features = extract_features(matrix_df, column_id='id', column_sort='time')
    return features

def main():
    # Get the DataFrame from your method
    matrix_df = GradientSet.all()[8].mat_df()
    GradientSet.all()[8].get_tsfresh_data()

    # Extract features
    features = extract_tsfresh_features(matrix_df)
    sns.heatmap(features.corr(), annot=True, cmap='coolwarm')
    import pdb
    pdb.set_trace()
    print(features)

if __name__ == '__main__':
    
    main()