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
        'r_1': 'r1.txt'
    }
    
    data = {}
    for key, filename in files.items():
        file_path = os.path.join(folder_path, filename)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Expected file not found: {file_path}")
        df = read_txt(file_path)
        df = add_datetime_index(df)        
        data[key] = df

    density, alpha_1, alpha_2, r_1 = data['density'], data['alpha_1'], data['alpha_2'], data['r_1']

    # Create shared full hourly index
    full_index = pd.date_range(
        start=density.index.min(),
        end=density.index.max(),
        freq='h'
    )

    dfs = [density, alpha_1, alpha_2, r_1]
    dfs_interpolated = [df.reindex(full_index).interpolate() for df in dfs]

    # Check time consistency
    ok, msg = check_time(*dfs_interpolated)
    print(msg)
    if not ok:
        raise ValueError("Time check failed.")

    # Save to file if path provided
    if save_path is not None:
        to_save = {
            'density': dfs_interpolated[0],
            'alpha_1': dfs_interpolated[1],
            'alpha_2': dfs_interpolated[2],
            'r_1': dfs_interpolated[3]
        }
        pd.to_pickle(to_save, save_path)
        print(f"Preprocessed data saved to {save_path}")

    return tuple(dfs_interpolated)