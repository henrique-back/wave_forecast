import pandas as pd

def check_time(*dfs):
    """
    Checks whether all input DataFrames:
    - Have datetime indexes
    - Share the same index
    - Are hourly without missing timestamps

    Parameters:
        *dfs: Arbitrary number of pandas DataFrames

    Returns:
        bool: True if all conditions are met, False otherwise
        str: Explanation of the result
    """
    if not dfs:
        return False, "No DataFrames provided."

    # Get reference index
    ref_index = dfs[0].index

    # Check if reference index is datetime
    if not pd.api.types.is_datetime64_any_dtype(ref_index):
        return False, "First DataFrame index is not datetime."

    # Check if all indexes are equal and datetime
    for i, df in enumerate(dfs[1:], start=2):
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            return False, f"DataFrame {i} index is not datetime."
        if not df.index.equals(ref_index):
            return False, f"DataFrame {i} index does not match the first DataFrame."

    # Check if the reference index is hourly and without gaps
    expected_range = pd.date_range(start=ref_index.min(), end=ref_index.max(), freq='H')
    if not ref_index.equals(expected_range):
        missing = expected_range.difference(ref_index)
        return False, f"Index is not continuous hourly; {len(missing)} missing timestamps."

    return True, "All DataFrames have matching, continuous hourly datetime indexes."
