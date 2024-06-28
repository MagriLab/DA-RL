import ml_collections
import numpy as np

def get_config():
    config = ml_collections.ConfigDict()
    config.nu = 0.08
    config.actuator_scale = 0.1
    config.burn_in = 1000
    config.actuator_locs = ((2 * np.pi)/8) * np.arange(8)
    config.sensor_locs = ((2 * np.pi)/16) * np.arange(16)
    config.target = 'e0'
    return config
