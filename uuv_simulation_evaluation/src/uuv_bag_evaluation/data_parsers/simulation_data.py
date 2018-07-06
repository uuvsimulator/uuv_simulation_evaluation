# Copyright (c) 2016 The UUV Simulator Authors.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np 
import logging
import sys


class SimulationData(object):
    LABEL = ""

    def __init__(self, topic_name=None, message_type=None, prefix=None):
        # Setting up the log
        self._logger = logging.getLogger(self.LABEL)
        if len(self._logger.handlers) == 0:
            out_hdlr = logging.StreamHandler(sys.stdout)
            out_hdlr.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
            out_hdlr.setLevel(logging.INFO)
            self._logger.addHandler(out_hdlr)
            self._logger.setLevel(logging.INFO)

        self._topic_name = topic_name
        self._message_type = message_type
        self._time = None
        self._prefix = prefix
        self._recorded_data = dict()
        self._output_dir = '/tmp'

    @staticmethod
    def get_all_parsers():
        return SimulationData.__subclasses__()

    @staticmethod
    def get_all_labels():
        return [parser.LABEL for parser in SimulationData.get_all_parsers()]
        
    def read_data(self, bag):
        raise NotImplementedError()

    def get_data(self, *args):
        raise NotImplementedError()

    def plot(self, output_dir):
        raise NotImplementedError()
    
    def get_data(self):
        return self._time, self._recorded_data    