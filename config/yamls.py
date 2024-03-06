import yaml

def read_yaml(file_path: str = './config/data_info.yaml') -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.

    Raises:
        FileNotFoundError: If the YAML file is not found at the specified path.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"YAML file not found at {file_path}")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")