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
from __future__ import print_function
import rospy
import numpy as np
import os
import matplotlib.pyplot as plt
from simulation_data import SimulationData


class SalinityData(SimulationData):
    LABEL = 'salinity_sensor'

    def __init__(self, bag):
        super(SalinityData, self).__init__()
        
        self._plot_configs = dict(salinity=dict(
                                    figsize=[12, 5],
                                    linewidth=2,
                                    label_fontsize=30,
                                    xlim=None,
                                    ylim=None,
                                    zlim=None,
                                    tick_labelsize=25,
                                    labelpad=10,
                                    legend=dict(
                                        loc='upper right',
                                        fontsize=25)))

        self._unit = None
        for x in bag.get_type_and_topic_info():
            for k in x:
                if 'uuv_sensor_plugins_ros_msgs/Salinity' in x[k][0]:
                    self._topic_name = k
                    self._logger.info('Particle salinity topic found <%s>' % k)
                    break
            if self._topic_name is not None:
                break

        self._time = list()

        try:
            self._recorded_data['salinity'] = list()
            for topic, msg, time in bag.read_messages(self._topic_name):
                    time = msg.header.stamp.to_sec()
                    self._time.append(time)
                    self._recorded_data['salinity'].append(msg.salinity)
                    if self._unit is None:
                        self._unit = msg.unit
            self._logger.info('%s=loaded' % self._topic_name)
        except Exception as e:
            self._logger.error('Error reading salinity topic, message=' + str(e))

    def plot(self, output_dir):
        if not os.path.isdir(output_dir):
            self._logger.error('Invalid output directory, dir=' + str(output_dir))
            raise Exception('Invalid output directory')

        fig = plt.figure(figsize=(self._plot_configs['salinity']['figsize'][0],
                                  self._plot_configs['salinity']['figsize'][1]))
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            
            ax = fig.add_subplot(111)

            ax.plot(self._time, self._recorded_data['salinity'], 'r', label=r'Salinity [%s]' % self._unit,
                    linewidth=self._plot_configs['salinity']['linewidth'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['salinity']['label_fontsize'])
            ax.set_ylabel(r'Salinity [%s]' % self._unit,
                          fontsize=self._plot_configs['salinity']['label_fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['salinity']['tick_labelsize'])

            ax.set_ylim([np.min(self._recorded_data['salinity']) - 0.1, 
                         np.max(self._recorded_data['salinity']) + 0.1])
            
            ax.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'salinity.pdf'))
            plt.close(fig)
            del fig

            if len(self._time) == 0:
                self._logger.error('No salinity information found')
                return
        except Exception as e:
            self._logger.error('Error while plotting salinity, message=' + str(e))
            plt.close(fig)
            del fig
