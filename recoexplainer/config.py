import yaml
from box import Box
import os

env = os.environ["ENVIRONMENT"]

with open("adv_ds_config.yml", "r") as yml_file:
    full_cfg = yaml.safe_load(yml_file)

cfg = Box({**full_cfg["base"], **full_cfg[env]}, default_box=True, default_box_attr=None)
cfg.environment = env
