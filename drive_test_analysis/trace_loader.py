"""Load CSV NEMO traces to Panda data-frames"""
import logging
import os, re
import pandas as pd

def _prepend_data_directory_location(data_directory_location,data_list):
    """Prepend the location of the data directory to each element of the data-file strings list"""
    # In place modfication thanks to x[0:]
    data_list[0:] = [data_directory_location+'/'+x for x in data_list[0:]]

def get_data_file_list(data_directory_location_list=None):
    """Returns a list with path of drive test files to load

    data_directory_location_list: list of directories containing the drive test files

    """
    logging.debug('Return data-file list')
    drive_test_data_list = list()
    for data_directory_location in data_directory_location_list:
        drive_test_data_list += [os.path.join(data_directory_location,filename) for filename in os.listdir(data_directory_location) \
                                 if re.search(r'\.csv.gz$|\.csv$', filename, re.IGNORECASE)]
    return drive_test_data_list

def load_data_file(data_file_list,k=None):
    """Load each data-file in a Panda frame or the selected one"""
    logging.debug('Load data into pandas')
    # Need to add support for .csv only
    if k is None:
        data = [pd.read_csv(data_file,sep=';',compression='gzip',low_memory=False) for data_file in data_file_list]
    else:
        data = pd.read_csv(data_file_list[k],sep=';',compression='gzip',low_memory=False)
    return data

def print_list(file_list):
    for k,item in enumerate(file_list):
        print('{}: {}'.format(k,item))

def concat_pandas_data(data_list):
    """Concatenate the pandas data-frame or data_series"""
    logging.debug('Concatenate pandas data-structure')
    return pd.concat(data_list,ignore_index=True)

def main():
    # Will put some tests eventually
    pass

if __name__ == "__main__":
    main()
