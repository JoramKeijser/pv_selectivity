"""
Helper functions
"""
import os
import pandas as pd
import warnings

def write_excel(save_to, dataframes):
    """
    Write dictionary of dataframes to excel
    Arguments:
        save_to (str): path where to save
        dataframes (dict): dictionary of dataframes
    Returns:
        None
    """
    if os.path.isfile(save_to):
        # File exists, append and overwrite
        mode = 'a'
        if_sheet_exists='replace'
        print("Append & ovewrite data to", save_to)
    else:
        mode = 'w'
        if_sheet_exists=None
        print("Save data to", save_to)
    with pd.ExcelWriter(save_to, mode=mode, if_sheet_exists=if_sheet_exists) as writer:
        for name, df in dataframes.items():
            if len(df) == 0:
                warnings.warn(f"Sheet {name} is empty")
            df.to_excel(writer, sheet_name=name)
