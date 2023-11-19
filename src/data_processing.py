import argparse
import pandas as pd
import numpy as np
import datetime

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

    # Checking if there's missing values and replacing them with mean value of preceeding and following samples
    if df.isna().sum().sum() > 0:
        df.interpolate(method='linear', limit_direction='both', inplace=True)

    # Imputing outliers with mean value
    for col in df.columns:
        outliers = identify_outliers(df[col])
        df.loc[outliers, col] = df[col].mean()


    return df

def preprocess_data(df):
    # TODO: Generate new features, transform existing features, resampling, etc.

    return df_processed

def save_data(df, output_file):
    # TODO: Save processed data to a CSV file
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