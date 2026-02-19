import numpy as np
import os
import json
from genericpath import isfile


def save_npArray(
        array: np.array,
        folder_path: str,
        file_name: str,
        comment: str = '') -> None:
    """Store a np.array in 'file_path/file_name'

    Parameters
    ----------
    array: np.array, 1D or 2D array to be saved
    folder_path: str, path to folder in which file is stored
    file_name: str, name of file,
    comment: str, add description of file to header
    """
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, f'{file_name}')
    np.savetxt(file_path, array, header=comment)
    print(f'Saved: {file_path}')


def load_npArray(folder_path: str, file_name: str) -> np.array:
    """Load a np.array from 'file_path/file_name'

    Parameters
    ----------
    folder_path: str, path to folder in which file is stored
    file_name: str, name of file

    Return
    ------
    loaded array as np.array
    """
    file_path = os.path.join(folder_path, f'{file_name}')
    if not isfile(file_path):
        raise FileNotFoundError(
            f"File {file_path} does not exist. Check path and file name")
    else:
        load_array = np.loadtxt(file_path)
        print(f'Loaded: {file_path}')
        return load_array


def save_json(output_dic: dict, output_path: str) -> str:
    """Save a dictionary as json file in 'output_path/preprocessed_data.json'.
    If this file already exists, add a counter to the file name
    to avoid overwriting an exisiting file.

    Parameters
    ----------
    output_dic: dict, dictionary to be saved as json file
    output_path: str, path to folder in which file is stored

    Return
    ------
    output_file: str, path to saved json file
    """
    if isfile(output_path+'/preprocessed_data.json'):
        print(("Preprocessed data file already exists! "
               "Adjusting output file!"))

    # Make sure no file is overwritten
    safety_file = output_path+'/preprocessed_data.json'
    safety_counter = 1
    while isfile(safety_file):
        safety_file = (
            f"{output_path}/preprocessed_data_{safety_counter}.json"
        )
        safety_counter += 1
    if safety_counter > 1:
        output_file = safety_file
    else:
        output_file = output_path+'/preprocessed_data.json'
    print('Saving preprocessed data to '+output_file)

    with open(output_file, 'w') as f:
        json.dump(output_dic, f)

    return output_file
