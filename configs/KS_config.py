import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.N = 64
    config.nu = 0.08
    config.actuator_scale = 0.4
    config.burn_in = 1000
    config.actuator_locs = ((2 * np.pi) / 4) * np.arange(4)
    config.sensor_locs = ((2 * np.pi) / 3) * np.arange(3)
    config.target = "e1"
    config.noise_stddev = 0.0
    config.frame_skip = 1
    return config
