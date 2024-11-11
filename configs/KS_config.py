import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.N = 64
    config.nu = 0.03
    config.actuator_scale = 0.3
    config.actuator_loss_weight = 1.0
    config.burn_in = 1000
    config.actuator_locs = ((2 * np.pi) / 9) * np.arange(9)
    config.sensor_locs = ((2 * np.pi) / 6) * np.arange(6)
    config.target = "e0"
    config.frame_skip = 1
    return config
