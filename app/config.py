import logging
import copy
import yaml
from pprint import pformat

# base configuration dict (overidable)
base_config = {

    "video" : {
        "output_path" : "./output",
        "input_path"  : None,
        "dataset_path": None,
        "start_ms"    : None,
        "end_ms"      : None,
        "cvat"        : False,
        "perf"        : False
    },
    
    "image" : {
        "output_path" : "./output"
    },

    "detect" : {
        "model_path" : "./models/yolo11l",
        "tile" : None,
        "conf" : 0.5
    },

    "track" : {
        "tracker"       : None,
        "embedder"      : None,
        "embedder_wts"  : None
    }
}

def cfg_update_from_yaml(config, yaml_path, copy_cfg = True):
    
    logging.info(f'cfg_update_from_yaml from {yaml_path}')

    cfg = copy.deepcopy(config) if copy_cfg else config
    
    try:
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
            logging.debug(f'yaml_data: {pformat(yaml_data)}')
        
        cfg = cfg_update(cfg,yaml_data,copy_cfg = False )

        # cfg.__dict__.update(yaml_data)  # Override values with YAML
    except Exception as e:
        logging.error(f'cfg_update_from_yaml- error: {str(e)}')

    return cfg


def cfg_update(config, override, config_key = '', copy_cfg = True) :
    '''
    Recursively update the base config with dict values 
    '''
    cfg = copy.deepcopy(base_config) if copy_cfg else config

    for key, value in override.items():

        key_name = config_key +'.'+key
        if key not in cfg:
            logging.warning(f'⚠️ cfg_update - unknownn config key : {key_name}')

        if isinstance(value, dict):
            if key not in cfg: cfg[key] = {}
            cfg_update(cfg[key], value, config_key=key_name, copy_cfg = False) # do not deepcopy nested dicts again
        else:
            cfg[key] = value

    return cfg


def cfg_get_base_config():
    return base_config
    
def cfg_init():
    cfg_update_from_yaml(base_config,'./config/default.yaml', copy_cfg = False)
    logging.info(f'base_config: {pformat(base_config)}')

cfg_init()