import os


def data_directory():
    p=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'dataset'))
    assert os.path.exists(p), "{} does not exist".format(p)
    return p


def root_directory():
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "src"))
    assert os.path.exists(p), "{} does not exist".format(p)
    return p
def result_directory():
    return ensure_directory_exists(os.path.join(root_directory(), "results"))


def plots_directory():
    return ensure_directory_exists(os.path.join(result_directory(), "plots"))


def ensure_directory_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder


if __name__ == "__main__":
    print(root_directory())
    print(data_directory())
    print(result_directory())
    print(plots_directory())
