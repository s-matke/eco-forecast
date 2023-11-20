import argparse
import pandas as pd
import numpy as np
import datetime
from utils import region_code_dict
from sklearn.utils import resample

RESAMPLE = False
OUTLIERS = False

def load_data(file_path):
    # TODO: Load data from CSV file
    df = pd.read_csv(file_path)
    return df

# find outliers statistically using percentiles
def identify_outliers(data):

    # q1 q3 percentiles
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)

    IQR = Q3 - Q1

    # lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = (data < lower_bound) | (data > upper_bound)

    return outliers


def clean_data(df):
    # TODO: Handle missing values, outliers, etc.
    df['StartTime'] = pd.to_datetime(df['StartTime'])
    df['month'] = df['StartTime'].dt.month

    # Checking if there's missing values and replacing them with mean value of preceeding and following samples
    if df.isna().sum().sum() > 0:
        df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Imputing outliers with mean value
    # TODO: Find outliers per month instead of looking at the entire year
    # Also through EDA it was denoted that there're no outliers which diverge from the norm
    # The outliers here are just additional energy generated in some cases, or less energy generated in other cases
    # As such we could skip over the part with handling outliers
    if OUTLIERS:
        for col in df.columns:
            outliers = identify_outliers(df[col])
            df.loc[outliers, col] = np.nan
            df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    return df

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.
    
    if 'Unnamed: 0' in list(df.columns):
        df.drop('Unnamed: 0', axis=1, inplace=True)

    region_codes = [col.split('_')[2] for col in df.columns if 'green_energy' in col]

    # Calculate surplus for each region so we can find the country with highest surplus and add it as our label feature
    for region_code in region_codes:
        df[f'surplus_{region_code}'] = np.abs(df[f'green_energy_{region_code}'] - df[f'{region_code}_Load'])
    
    df['label'] = df.apply(lambda row: region_codes[row[[f'surplus_{country}' for country in region_codes]].values.argmax()], axis=1)
    df['label'].replace(region_code_dict, inplace=True)

    # # Dropping the features for surplus
    # for region_code in region_codes:
    #     df.drop(f'surplus_{region_code}', axis=1, inplace=True)

    # Not the greatest way of resampling our data, but will do the job for now
    # Losing a lot of information by resampling, so I've set it as an optional feature for the sake of testing
    if RESAMPLE:
        label_counts = dict(df['label'].value_counts())
        highest_count_label = max(label_counts, key=label_counts.get)
        lowest_count_label = min(label_counts, key=label_counts.get)

        # Downsampling the label that occurs the most
        df_temp = df.loc[df['label'] == highest_count_label, :]
        df.drop(df.loc[df['label'] == highest_count_label, :].index, inplace=True)

        new_highest_label_count = max(df['label'].value_counts())

        df_temp = resample(df_temp, replace=True, n_samples=new_highest_label_count, random_state=42)

        df = pd.concat([df, df_temp])

        # Upsampling the label that occurs the least
        df_temp = df.loc[df['label'] == lowest_count_label, :]
        df.drop(df.loc[df['label'] == lowest_count_label, :].index, inplace=True)

        new_lowest_label_count = min(df['label'].value_counts())

        df_temp = resample(df_temp, replace=True, n_samples=new_lowest_label_count, random_state=42)

        df = pd.concat([df, df_temp])
            
    return df    

def save_data(df, output_file):
    try:
        df.to_csv(output_file, index=False)
    except:
        print(f"Error: Something went wrong trying to save the file at the following path: {output_file}")
    pass

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data processing script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--input_file',
        type=str,
        default='data/raw_data.csv',
        help='Path to the raw data file to process'
    )
    parser.add_argument(
        '--output_file', 
        type=str, 
        default='data/processed_data.csv', 
        help='Path to save the processed data'
    )
    return parser.parse_args()

def main(input_file, output_file):
    df = load_data(input_file)
    df_clean = clean_data(df)
    df_processed = preprocess_data(df_clean)
    save_data(df_processed, output_file)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.input_file, args.output_file)