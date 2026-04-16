import os
import pandas as pd
from utils import read_txt, add_datetime_index, check_time

def data_processing(folder_path, save_path=None):
    """
    Reads and processes buoy data files, reindexes to full hourly index and interpolates.
    
    Parameters:
        folder_path (str): Path to folder containing the .txt files.
        
    Returns:
        tuple: (density, alpha_1, alpha_2, r_1) DataFrames after reindexing & interpolation.
    """
    files = {
        'density': 'density.txt',
        'alpha_1': 'alpha1.txt',
        'alpha_2': 'alpha2.txt',
        'r_1': 'r1.txt',
    }
    
    data = {}
    for key, filename in files.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Expected file not found: {file_path}")
        df = read_txt(file_path)
        df = add_datetime_index(df)        
        df = df.sort_index()
        df = df[~df.index.duplicated()]

        data[key] = df

    cols = data['density'].columns
    for name, df in data.items():
        if not df.columns.equals(cols):
            raise ValueError(f"Column mismatch in {name}")
        
    # Create shared full hourly index  
    start = min(df.index.min() for df in data.values())
    end = max(df.index.max() for df in data.values())
    full_index = pd.date_range(start=start, end=end, freq='h')

    dfs_interpolated = [
        df.reindex(full_index).interpolate(method='time', limit_direction='both')
        for df in data.values()
    ]
    
    # Check time consistency
    ok, msg = check_time(*dfs_interpolated)
    print(msg)
    if not ok:
        raise ValueError("Time check failed.")

    # Save to file if path provided
    if save_path is not None:
        pd.to_pickle(tuple(dfs_interpolated), save_path)
        print(f"Preprocessed data saved to {save_path}")

    return tuple(dfs_interpolated)