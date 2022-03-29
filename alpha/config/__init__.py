import os
import cfg4py


def get_config_dir():
    if os.getenv("__cfg4py_server_role__", "DEV"):
        return os.path.dirname(__file__)
    else:
        return os.path.expanduser("~/.zillionare/alpha")
