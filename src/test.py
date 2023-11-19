import os

def load_merge_save_raw_dataframes(data_dir_path):
    merges = []
    df_merged = None

    region_sub_dirs = [name for name in os.listdir(data_dir_path) if os.path.isdir(os.path.join(data_dir_path, name))]
    print(region_sub_dirs)

    for region_dir in region_sub_dirs:
        pass


load_merge_save_raw_dataframes('./data')