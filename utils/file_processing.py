import pickle


def unpickle_file(file_name):
    file = open(file_name, "rb")
    data = pickle.load(file)
    return data, file


def pickle_file(file_name, data):
    file = open(file_name, "wb")
    pickle.dump(data, file)
    file.close()
    return
