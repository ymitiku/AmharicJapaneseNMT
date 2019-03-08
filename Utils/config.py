import yaml
import os

def load_config(config_file):
    if not os.path.exists(config_file):
        raise FileNotFoundError("Config file:'%s' is not found" % config_file)
    with open(config_file) as c_file:
        config = yaml.safe_load(c_file)
    return config
