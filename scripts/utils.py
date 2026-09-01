'''
This file contains utility helpers for experiment scripts, handles config loading and prediction serialization
'''

import yaml
import pandas as pd

def load_config(config_path):
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"configuration file not found at {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"error parsing yaml config file:{e}")
    except Exception as e:
        raise ValueError(f"unexpected error while loading config:{e}")

def save_predictions(predictions, output_path):
    try:
        df = pd.DataFrame(predictions)
        df.to_csv(output_path, index=False)
    except Exception as e:
        raise IOError(f"failed to save predictions to {output_path}:{e}")
