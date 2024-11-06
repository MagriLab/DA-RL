import pickle
import h5py
import yaml


def write_h5(path, data):
    """Write dictionary to a .h5 file"""
    hf = h5py.File(path, "w")  # with hf
    print(path, flush=True)
    for k, v in data.items():
        hf.create_dataset(k, data=v)
    hf.close()


def read_h5(path):
    """Read from dictionary in a .h5 file

    Args:
        path: file path to data
    Returns:
        data_dictionary: dictionary that contains the items in the h5 file
    """
    data_dict = {}
    with h5py.File(path, "r") as hf:
        for k in hf.keys():
            data_dict[k] = hf.get(k)[()]
    return data_dict


def unpickle_file(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    return data, file


def pickle_file(file_name, data):
    file = open(file_name, "wb")
    pickle.dump(data, file)
    file.close()
    return


def save_config(experiment_path, config):
    with open(experiment_path / "config.yml", "w") as f:
        config.to_yaml(stream=f)


def load_config(experiment_path):
    with open(experiment_path / "config.yml", "r") as file:
        config = yaml.unsafe_load(file)
        return config
