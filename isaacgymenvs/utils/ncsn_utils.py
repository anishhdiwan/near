import argparse

def dict2namespace(config):
    """Convert a disctionary (typically containing config params) to a namespace structure (https://tedboy.github.io/python_stdlib/generated/generated/argparse.Namespace.html#argparse.Namespace)

    Args:
        config (dict): dictionary of configs params
    """
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace