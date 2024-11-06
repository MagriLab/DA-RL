import os
from jax.lib import xla_bridge
from typing import Union


def on_which_platform():
    print(xla_bridge.get_backend().platform)


def set_gpu(gpu_id: Union[int, str], memory_fraction: float = 1.0):
    """Set which gpu to use and the memory fraction using CUDA_VISIBLE_DEVICES"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(memory_fraction)
