import logging.config
import json
import os

def setup_logging(config_path="logger.json", loglevel=logging.WARNING):
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        logging.config.dictConfig(config)
        logging.getLogger().setLevel(loglevel)
    else:
        logging.basicConfig(level=loglevel)
