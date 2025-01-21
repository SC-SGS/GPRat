import json

config_file = "./config.json"

with open(config_file) as cf:
    config = json.load(cf)


def get_config_file():
    return config_file


def get_config():
    return config
