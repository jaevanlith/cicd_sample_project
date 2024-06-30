import pandas as pd

def write_to_csv(dataframe, filename):
    '''
    Write a dataframe to a csv file.
    '''
    assert isinstance(dataframe, pd.DataFrame), 'dataframe must be a pandas DataFrame'
    assert filename.endswith('.csv'), 'filename must end with .csv'

    dataframe.to_csv(filename)
    