# data_utils.py

import pickle
import json
import os
import zipfile


def save_data(filepath, data, current_name="ttmp_chunk", chunk_size=500):
    """
    Save data to a ZIP file in pickle format in chunks.
    
    Parameters:
    - filepath (str): The path where the ZIP file should be saved.
    - data: The data to save (a dictionary containing a list of dictionaries).
    - chunk_size (int): Number of items per chunk.
    """
    # Ensure that 'data' is a dictionary with a list of dictionaries
    if not isinstance(data, dict) or not all(isinstance(v, list) for v in data.values()):
        raise ValueError("Data should be a dictionary containing a list of dictionaries.")

    # Ensure the filepath has the correct .zip extension
    if not filepath.endswith('.zip'):
        filepath = filepath + '.zip'

    # Create a directory for temporary chunk files
    temp_dir ='./temp_chunks' # 
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    # Create a zip file for writing
    with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Iterate over the dictionary and save each chunk
        for key, dict_list in data.items():
            for i in range(0, len(dict_list), chunk_size):
                chunk = dict_list[i:i + chunk_size]
                chunk_filename = f"{key}_chunk_{i // chunk_size}_{current_name}.pkl"
                chunk_path = os.path.join(temp_dir, chunk_filename)
                
                # Save each chunk to the temporary directory
                with open(chunk_path, 'wb') as temp_file:
                    pickle.dump({key: chunk}, temp_file)
                
                # Add the temporary file to the zip archive
                zip_file.write(chunk_path, arcname=chunk_filename)
                
                # Remove the temporary file after adding it to the zip archive
                os.remove(chunk_path)


def load_data(filepath):
    """
    Load data from a ZIP file containing pickled chunks.
    
    Parameters:
    - filepath (str): The path to the ZIP file from which data should be loaded.
    
    Returns:
    - data: The reconstructed data (a dictionary containing a list of dictionaries).
    """
    # Dictionary to hold the reconstructed data
    reconstructed_data = {}

    # Open the ZIP file for reading
    with zipfile.ZipFile(filepath + '.zip', 'r') as zip_file:
        # Iterate over each file in the ZIP archive
        for file_name in zip_file.namelist():
            # Extract the file content to memory
            with zip_file.open(file_name) as file:
                # Load the data chunk from the file
                chunk_data = pickle.load(file)
                
                # Merge the chunk into the reconstructed data dictionary
                for key, chunk in chunk_data.items():
                    if key not in reconstructed_data:
                        reconstructed_data[key] = []
                    reconstructed_data[key].extend(chunk)
    
    return reconstructed_data

def save_parameters(base_path, filename, params):
    """
    Save simulation parameters to a JSON file.
    
    Parameters:
    - base_path (str): The base path to save the file.
    - filename (str): The name of the file.
    - params (dict): The parameters to save.
    """
    with open(os.path.join(base_path, filename), 'w') as file:
        json.dump(params, file)

def load_parameters(filepath):
    """
    Load simulation parameters from a JSON file.
    
    Parameters:
    - filepath (str): The path to the JSON file.
    
    Returns:
    - The loaded parameters.
    """
    with open(filepath, 'r') as file:
        return json.load(file)
