#!/usr/bin/env python
# coding=utf-8
from __future__ import print_function

import json
import os
import sys

CONFIG_FILE = "config.json"


class Config(object):
    def __init__(self):
        config_path = os.path.abspath(CONFIG_FILE)
        if not os.path.isfile(config_path):
            print('Error: can\'t find config file {0}.'.format(config_path))
            sys.exit(1)

        # noinspection PyBroadException
        try:
            with open(config_path, 'r') as f:
                parameters = json.load(f)

            self.__dict__ = parameters
        except Exception:
            print('Error: can\'t parse config file {0}.'.format(config_path))
            sys.exit(1)

config = Config()
