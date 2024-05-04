import yaml


def readYaml(yaml_path: str):
    with open(yaml_path, 'r') as stream:
        try:
            yaml_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return yaml_data


def saveYaml(yaml_path: str, data: dict):
    with open(yaml_path, 'a') as stream:
        try:
            yaml.dump(data, stream)
        except yaml.YAMLError as exc:
            print(exc)
