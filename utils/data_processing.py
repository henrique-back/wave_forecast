import os
import numpy as np
import pandas as pd
from utils import read_txt, add_datetime_index, check_time

# NDBC standard meteorological ("stdmet") missing-value sentinels for the
# columns we read. Must be masked to NaN before any interpolation.
_WIND_SENTINELS = {'WDIR': 999, 'WSPD': 99.0}

# alpha_1/alpha_2 (mean/principal wave direction per frequency bin, degrees)
# are circular, same as WDIR above.
_CIRCULAR_DEGREE_KEYS = {'alpha_1', 'alpha_2'}


def _interpolate_circular_degrees(df, full_index):
    """Time-interpolate a DataFrame of angles (degrees, circular) per column.

    Same wraparound problem as WDIR in process_wind(): linearly interpolating
    a raw angle across a gap (e.g. 350deg -> 10deg) passes through 180deg
    instead of through 0deg. Decomposing into sin/cos, interpolating those,
    and recombining via atan2 keeps the interpolation on the correct (short)
    arc, per frequency bin.
    """
    theta = np.radians(df)
    sin_part = np.sin(theta).reindex(full_index).interpolate(method='time', limit_direction='both')
    cos_part = np.cos(theta).reindex(full_index).interpolate(method='time', limit_direction='both')
    return np.degrees(np.arctan2(sin_part, cos_part)) % 360


def process_wind(folder_path, filename='wind.txt'):
    """
    Reads a buoy NDBC stdmet .txt file and derives wind_u/wind_v.

    WDIR (degrees, direction wind is coming FROM) is circular — linearly
    time-interpolating the raw angle across a data gap (e.g. 350deg -> 10deg)
    would pass through 180deg and produce a wrong intermediate direction.
    u/v components are computed here, BEFORE any time-reindexing/interpolation
    happens in data_processing(), so that step interpolates two continuous
    scalars instead of a wrapping angle.

    Returns:
        pd.DataFrame with columns ['wind_u', 'wind_v'], datetime-indexed,
        NOT yet reindexed to the shared hourly grid.
    """
    file_path = os.path.join(folder_path, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Expected file not found: {file_path}")

    df = read_txt(file_path)
    df = add_datetime_index(df)
    df = df.sort_index()
    df = df[~df.index.duplicated()]

    df = df[['WDIR', 'WSPD']].copy()
    for col, sentinel in _WIND_SENTINELS.items():
        df[col] = df[col].where(df[col] != sentinel)

    theta = np.radians(df['WDIR'])
    wind = pd.DataFrame(index=df.index)
    wind['wind_u'] = -df['WSPD'] * np.sin(theta)
    wind['wind_v'] = -df['WSPD'] * np.cos(theta)

    return wind


def data_processing(folder_path, save_path=None):
    """
    Reads and processes buoy data files, reindexes to full hourly index and interpolates.

    Parameters:
        folder_path (str): Path to folder containing the .txt files.

    Returns:
        tuple: (density, alpha_1, alpha_2, r_1, r_2, wind) DataFrames after
        reindexing & interpolation. wind has columns ['wind_u', 'wind_v'].
    """
    files = {
        'density': 'density.txt',
        'alpha_1': 'alpha1.txt',
        'alpha_2': 'alpha2.txt',
        'r_1': 'r1.txt',
        'r_2': 'r2.txt',
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

    # Wind has a structurally different set of columns (2 named scalars, not
    # frequency bins) so it is kept out of the column-equality check above,
    # but shares the same reindex/interpolate/check_time treatment below.
    data['wind'] = process_wind(folder_path)

    # Create shared full hourly index
    start = min(df.index.min() for df in data.values())
    end = max(df.index.max() for df in data.values())
    full_index = pd.date_range(start=start, end=end, freq='h')

    dfs_interpolated = [
        _interpolate_circular_degrees(df, full_index) if key in _CIRCULAR_DEGREE_KEYS
        else df.reindex(full_index).interpolate(method='time', limit_direction='both')
        for key, df in data.items()
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