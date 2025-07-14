import pandas as pd

def add_datetime_index(df, year='YY', month='MM', day='DD', hour='hh', minute='mm', drop_original=True):
    """
    Adds a datetime index to a DataFrame using separate year/month/day/hour/minute columns.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        year, month, day, hour, minute (str): Column names for date-time parts.
        drop_original (bool): If True, drops the original time columns after conversion.

    Returns:
        pd.DataFrame: DataFrame indexed by floored hourly datetime.
    """
    datetime_cols = df[[year, month, day, hour, minute]].rename(columns={
        year: 'year', month: 'month', day: 'day', hour: 'hour', minute: 'minute'
    })

    df['datetime'] = pd.to_datetime(datetime_cols)
    df = df.set_index('datetime').sort_index()
    df.index = df.index.floor('h')

    if drop_original:
        df = df.drop(columns=[year, month, day, hour, minute])

    return df
