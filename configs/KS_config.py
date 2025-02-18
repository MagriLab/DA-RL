import ml_collections
import numpy as np


def get_config():
    config = ml_collections.ConfigDict()
    config.N = 64
    config.nu = 0.08
    config.actuator_scale = 0.1
    config.actuator_loss_weight = 0.2
    config.burn_in = 1000
    config.actuator_locs = ((2 * np.pi) / 8) * np.arange(8)
    config.sensor_locs = ((2 * np.pi) / 64) * np.arange(64)
    config.target = "e0"
    config.frame_skip = 1
    return config
