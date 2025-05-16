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
    cfg = copy.deepcopy(config) if copy_cfg else config

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


def print_config(config):
    """
    Print the configuration in a hierarchical, readable format.
    
    Args:
        base_config (dict): Configuration dictionary to print
    """
    if not logging.getLogger().isEnabledFor(logging.INFO) : 
        return

    logging.info("Configuration:")
    logging.info("-" * 40)
    
    for section, settings in base_config.items():
        logging.info(f"[{section.upper()} SECTION]")
        
        # Handle empty or None sections
        if not settings:
            logging.info("  (No settings)")
            continue
        
        # Print each setting with alignment
        max_key_length = max(len(str(key)) for key in settings.keys())
        for key, value in settings.items():
            # Format the value representation
            if value is None:
                formatted_value = "(Not Set)"
            elif isinstance(value, bool):
                formatted_value = str(value)
            elif isinstance(value, (int, float)):
                formatted_value = str(value)
            else:
                formatted_value = f'"{value}"'
            
            # Align the output
            logging.info(f"  {key:{max_key_length}} : {formatted_value}")
    
    logging.info("-" * 40)


def cfg_get_base_config(copy_cfg = True):
    cfg = copy.deepcopy(base_config) if copy_cfg else base_config
    return cfg
    
def cfg_init():
    cfg_update_from_yaml(base_config,'./config/default.yaml', copy_cfg = False)
    print_config(base_config)


cfg_init()
