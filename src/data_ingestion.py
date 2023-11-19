"""
TODO:
    - Vratiti nazad podrazumevane vrednosti za parametre metoda get_load_* i get_gen_* (med prio)
    - Token ucitavati putem .env fajla (low prio)
    - 

"""
import argparse
import datetime
import pandas as pd
import numpy as np
from utils import perform_get_request, xml_to_load_dataframe, xml_to_gen_data, generate_region_subdirectories
import os
from functools import reduce

green_energy_sources = {
    'B01': 'Biomass',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore'
}

def get_load_data_from_entsoe(regions, periodStart='202201010000', periodEnd='202201012359', output_path='./data'):
    
    # TODO: There is a period range limit of 1 year for this API. Process in 1 year chunks if needed
    
    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    # Refer to https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_documenttype
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A65',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():

        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
    
        # Use the requests library to get data from the API for the specified time range
        response_content = perform_get_request(url, params)

        # Response content is a string of XML data
        df = xml_to_load_dataframe(response_content)

        # Save the DataFrame to a CSV file
        df.to_csv(f'{output_path}/{region}/load_{region}.csv', index=False)
       
    return

def get_gen_data_from_entsoe(regions, periodStart='202201010000', periodEnd='202201012359', output_path='./data'):
    
    # TODO: There is a period range limit of 1 day for this API. Process in 1 day chunks if needed

    # URL of the RESTful API
    url = 'https://web-api.tp.entsoe.eu/api'

    # General parameters for the API
    params = {
        'securityToken': '1d9cd4bd-f8aa-476c-8cc1-3442dc91506d',
        'documentType': 'A75',
        'processType': 'A16',
        'outBiddingZone_Domain': 'FILL_IN', # used for Load data
        'in_Domain': 'FILL_IN', # used for Generation data
        'periodStart': periodStart, # in the format YYYYMMDDHHMM
        'periodEnd': periodEnd # in the format YYYYMMDDHHMM
    }

    # Loop through the regions and get data for each region
    for region, area_code in regions.items():

        print(f'Fetching data for {region}...')
        params['outBiddingZone_Domain'] = area_code
        params['in_Domain'] = area_code
    
        # Use the requests library to get data from the API for the specified time range
        response_content = perform_get_request(url, params)

        # Response content is a string of XML data
        dfs = xml_to_gen_data(response_content)

        # Save the dfs to CSV files
        for psr_type, df in dfs.items():
            # Save the DataFrame to a CSV file
            df.to_csv(f'{output_path}/{region}/gen_{region}_{psr_type}.csv', index=False)
    
    return

def fill_missing_hours(df, start_time, end_time, frequency, df_type):
    start_time = pd.Timestamp(start_time)
    end_time = pd.Timestamp(end_time)

    min_start_time = df['StartTime'].min()
    max_start_time = df['StartTime'].max()

    if min_start_time > start_time:
        min_start_time = start_time.replace(minute=0)
    else:
        min_start_time = min_start_time.replace(minute=0)
    
    if max_start_time < end_time:
        max_start_time = end_time.replace(minute=0)
    else:
        max_start_time = max_start_time.replace(minutes=0)

    # creating dataframe that has all the days accounted for
    complete_range = pd.date_range(start=min_start_time, end=max_start_time, freq=f'{int(frequency)}T')
    complete_df = pd.DataFrame({'StartTime': complete_range})

    # Merge the original dataframe with the complete dataframe
    merged_df = pd.merge(complete_df, df, on='StartTime', how='left')

    # fill missing values with 0
    merged_df[df_type] = merged_df[df_type].fillna(0)
    
    return merged_df

def transform_dataframe(start_time, end_time, file_path):
    df = pd.read_csv(file_path)

    df.drop_duplicates('StartTime', keep='first', inplace=True)
    df.drop(['EndTime', 'AreaID', 'UnitName'], axis=1, inplace=True)

    # transform time values from string into datetime object 
    df['StartTime'] = df['StartTime'].apply(lambda x: datetime.datetime.strptime(x[:-1], "%Y-%m-%dT%H:%M%z"))
    df['StartTime'] = pd.to_datetime(df['StartTime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M')))

    # True if data is recorded in 15-minute marks
    short_interval = False

    time_diff = df['StartTime'].diff()
    unique_diffs = time_diff.unique()

    extracted_diffs_minutes = [diff.total_seconds() / 60 for diff in unique_diffs]
    filtered_list = [value for value in extracted_diffs_minutes if not np.isnan(value) and value > 0.0]
    count_req = 0

    if min(filtered_list) < 60.0:
        count_req = np.floor(60.0 / min(filtered_list))
        short_interval = True
        freq = min(filtered_list)
    else:
        freq = 60.0
    
    if 'quantity' in list(df.columns):
        df = fill_missing_hours(df, start_time, end_time, frequency=freq, df_type='quantity')
    else:
        df = fill_missing_hours(df, start_time, end_time, frequency=freq, df_type='Load')
    
    # Interpolate missing values and check if there's any data within dataframe we can use (if all values are 0 in the entire dataframe, we have no use of it)
    if 'PsrType' in list(df.columns):
        if df.loc[df['quantity'] == 0].shape[0] == df.shape[0]:
            return None
        
        df.drop(['PsrType'], axis=1, inplace=True)
        df['quantity'] = df['quantity'].replace(0, np.nan)
        df.interpolate(method='linear', limit_direction='both', inplace=True)
    else:
        if df.loc[df['Load'] == 0].shape[0] == df.shape[0]:
            return None
        df['Load'].replace(0, np.nan, inplace=True)
        df['Load'].interpolate(method='linear', limit_direction='both', inplace=True)

    # Add extra columns for year, month, day, hour
    df['hour'] = df['StartTime'].apply(lambda x: x.hour)
    df['day'] = df['StartTime'].apply(lambda x: x.day)
    df['month'] = df['StartTime'].apply(lambda x: x.month)
    df['year'] = df['StartTime'].apply(lambda x: x.year)

    if short_interval:
        hourly_counts = df.groupby(['year', 'month', 'day', 'hour']).size()

        missing_hours = hourly_counts[hourly_counts != count_req].index

        df = df[~df[['year', 'month', 'day', 'hour']].apply(tuple, axis=1).isin(missing_hours)]
        
        # Transform data from 15-minute interval format into 1-hour interval format per sample
        df = df.resample('H', on='StartTime', closed='left').sum(numeric_only=True).reset_index()
        
    df.drop(['year', 'month', 'day', 'hour'], axis=1, inplace=True)

    return df

def merge_green_energy_dataframes(dataframes, region_code):

    def custom_merge(df_left, df_right):
        merged_df = pd.merge(df_left, df_right, on=['StartTime'], how='outer', suffixes=('', '_other'))
        # Sum up the 'quantity' values from both dataframes
        merged_df['quantity'] = merged_df['quantity'] + merged_df['quantity_other']
        # Drop the redundant 'quantity_other' column
        merged_df = merged_df.drop('quantity_other', axis=1)
        return merged_df

    df_merged = reduce(custom_merge, dataframes)

    df_merged.rename(columns={'quantity': f'green_energy_{region_code}'}, inplace=True)

    return df_merged

def save_raw_data(merges, output_path):
    raw_data = reduce(lambda left, right: pd.merge(left, right, on=['StartTime'], how='outer'), merges)
    print("NANS: ", raw_data.isna().sum().sum())
    raw_data.to_csv(f'{output_path}/raw_data.csv')

def load_merge_save_raw_dataframes(start_time, end_time, data_dir_path):
    merges = []
    df_merged = None

    region_sub_dirs = [name for name in os.listdir(data_dir_path) if os.path.isdir(os.path.join(data_dir_path, name))]

    for region_dir in region_sub_dirs:

        region_dir_path = f'{data_dir_path}/{region_dir}/'
        region_files = os.listdir(region_dir_path)

        green_energy_dfs = []
        df_load = None

        for file_name in region_files:
            split_file_name = file_name.split('_')
            file_path = f'{region_dir_path}/{file_name}'

            if 'merged' in split_file_name:
                continue
            
            elif 'load' in split_file_name:
                df_load = transform_dataframe(start_time, end_time, file_path)
                df_load.rename(columns={'Load':f'{region_dir}_Load'}, inplace=True)
            else:
                psr_type = split_file_name[2].split('.')[0]

                if not psr_type in list(green_energy_sources.keys()):
                    continue

                df = transform_dataframe(start_time, end_time, file_path)

                if df is not None:
                    green_energy_dfs.append(df)
        
        if len(green_energy_dfs) != 0:
            df_merged = merge_green_energy_dataframes(green_energy_dfs, region_code=region_dir)
            df_merged = pd.merge(df_merged, df_load, how='outer')
        else:
            df_merged = df_load
        
        merges.append(df_merged)
    
    save_raw_data(merges, data_dir_path)    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data ingestion script for Energy Forecasting Hackathon')
    parser.add_argument(
        '--start_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 1, 1), 
        help='Start time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--end_time', 
        type=lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'), 
        default=datetime.datetime(2022, 1, 5), 
        help='End time for the data to download, format: YYYY-MM-DD'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        default='./data',
        help='Name of the output file'
    )
    return parser.parse_args()

def main(start_time, end_time, output_path):
    
    regions = {
        'HU': '10YHU-MAVIR----U',
        'IT': '10YIT-GRTN-----B',
        'PO': '10YPL-AREA-----S',
        'SP': '10YES-REE------0',
        'UK': '10Y1001A1001A92E',
        'DE': '10Y1001A1001A83F',
        'DK': '10Y1001A1001A65H',
        'SE': '10YSE-1--------K',
        'NE': '10YNL----------L',
    }

    # Generates subdirectories in data folder for each region
    try:
        generate_region_subdirectories(regions, output_path)
    except:
        raise Exception("Error: Couldn't create subdirectories.")

    # Transform start_time and end_time to the format required by the API: YYYYMMDDHHMM
    start_time = start_time.strftime('%Y%m%d%H%M')
    end_time = end_time.strftime('%Y%m%d%H%M')

    # Get Load data from ENTSO-E
    get_load_data_from_entsoe(regions, start_time, end_time, output_path)

    # Get Generation data from ENTSO-E
    get_gen_data_from_entsoe(regions, start_time, end_time, output_path)

    # Combine Generation data and Load data into single file
    load_merge_save_raw_dataframes(start_time, end_time, output_path)

if __name__ == "__main__":
    args = parse_arguments()
    main(args.start_time, args.end_time, args.output_path)