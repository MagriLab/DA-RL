import ml_collections
from ml_collections.config_dict import placeholder

def get_config():
    config = ml_collections.ConfigDict()

    config.mode = 'online'
    config.project = 'DA-RL'
    config.entity = placeholder(str)
    config.group = placeholder(str)
    config.job_type = placeholder(str)
    config.name = placeholder(str)
    config.tags = placeholder(tuple)
    
    return config
