import os


def get_config_dir():
    if os.getenv("__cfg4py_server_role__") == "DEV":
        path = os.path.dirname(__file__)
    else:
        path = os.path.expanduser("~/zillionare/alpha/config")

    return path
