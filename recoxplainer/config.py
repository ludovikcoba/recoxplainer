import yaml
from box import Box
import os

with open("configs/config.yml", "r") as yml_file:
    full_cfg = yaml.safe_load(yml_file)

cfg = Box({**full_cfg["base"]},
          default_box=True,
          default_box_attr=None)
