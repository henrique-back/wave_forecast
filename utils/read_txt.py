import pandas as pd

def read_txt(filepath):
    """
    Reads a buoy .txt file where the first commented line (starting with '#')
    contains the column names. Ignores any other commented lines.
    
    Parameters:
        filepath (str): Path to the .txt file.
        
    Returns:
        pd.DataFrame: DataFrame with parsed column names.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):
        if line.startswith('#'):
            header_line = line.lstrip('#').strip()
            columns = header_line.split()
            skiprows = idx + 1
            break
    else:
        raise ValueError("No commented header line starting with '#' found.")

    df = pd.read_csv(
        filepath,
        sep='\s+',
        names=columns,
        skiprows=skiprows,
        header=None,
        comment='#'  # ignore any remaining comment lines
    )

    return df
