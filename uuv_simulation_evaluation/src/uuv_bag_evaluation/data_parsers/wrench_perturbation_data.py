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

try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
except Exception, e:
    print('Cannot use Latex configuration with matplotlib, message=', str(e))

class WrenchPerturbationData(SimulationData):
    LABEL = 'wrench_perturbation'

    def __init__(self, bag):
        super(WrenchPerturbationData, self).__init__()

        self._plot_configs = dict(wrenches=dict(
                                    figsize=[12, 5],
                                    linewidth=2,
                                    label_fontsize=30,
                                    xlim=None,
                                    ylim=None,
                                    zlim=None,
                                    tick_labelsize=25,
                                    labelpad=10,
                                    legend=dict(loc='upper right',
                                                fontsize=18)))

        for x in bag.get_type_and_topic_info():
            for k in x:
                if 'wrench_perturbation' in k:
                    self._topic_name = k
                    self._logger.info('Wrench perturbation topic found <%s>', k)
                    break
            if self._topic_name is not None:
                break

        self._time = list()

        try:
            self._recorded_data['force'] = list()
            self._recorded_data['torque'] = list()
            for topic, msg, time in bag.read_messages(self._topic_name):
                    time = msg.header.stamp.to_sec()
                    self._time.append(time)
                    self._recorded_data['force'].append(
                        [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                    self._recorded_data['torque'].append(
                        [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
            self._logger.info('%s=loaded' % self._topic_name)
        except Exception, e:
            self._logger.error('Error retrieving wrench perturbation data from rosbag, message=' + str(e))
            self._recorded_data['force'] = None
            self._recorded_data['torque'] = None

    @property
    def disturbances(self):
        return self._time, self._recorded_data['force'], self._recorded_data['torque']

    def plot(self, output_dir):
        if not os.path.isdir(output_dir):
            self._logger.error('Invalid output directory, dir=' + str(output_dir))
            raise Exception('Invalid output directory')

        fig = plt.figure(figsize=(self._plot_configs['wrenches']['figsize'][0],
                                  self._plot_configs['wrenches']['figsize'][1]))
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            
            ax = fig.add_subplot(211)
            ax.plot(self._time, [f[0] for f in self._recorded_data['force']], 'r', label=r'$F_X$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(self._time, [f[1] for f in self._recorded_data['force']], 'g', label=r'$F_Y$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(self._time, [f[2] for f in self._recorded_data['force']], 'b', label=r'$F_Z$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['wrenches']['label_fontsize'])
            ax.set_ylabel('Force [N]',
                          fontsize=self._plot_configs['wrenches']['label_fontsize'])
            ax.legend(fancybox=True, framealpha=0.9,
                      loc=self._plot_configs['wrenches']['legend']['loc'],
                      fontsize=self._plot_configs['wrenches']['legend']['fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['wrenches']['tick_labelsize'])
            ax.grid(True)

            min_y = [np.min([v[0] for v in self._recorded_data['force']]),
                     np.min([v[1] for v in self._recorded_data['force']]),
                     np.min([v[2] for v in self._recorded_data['force']])]

            max_y = [np.max([v[0] for v in self._recorded_data['force']]),
                     np.max([v[1] for v in self._recorded_data['force']]),
                     np.max([v[2] for v in self._recorded_data['force']])]

            ax.set_ylim([np.min(min_y) * 1.1, np.max(max_y) * 1.1])
            ax.set_xlim(np.min(self._time), np.max(self._time))

            ax = fig.add_subplot(212)
            ax.plot(self._time, [f[0] for f in self._recorded_data['torque']], 'r', label=r'$\tau_X$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(self._time, [f[1] for f in self._recorded_data['torque']], 'g', label=r'$\tau_Y$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(self._time, [f[2] for f in self._recorded_data['torque']], 'b', label=r'$\tau_Z$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['wrenches']['label_fontsize'])
            ax.set_ylabel('Torque [Nm]',
                          fontsize=self._plot_configs['wrenches']['label_fontsize'])
            ax.legend(fancybox=True, framealpha=0.9,
                      loc=self._plot_configs['wrenches']['legend']['loc'],
                      fontsize=self._plot_configs['wrenches']['legend']['fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['wrenches']['tick_labelsize'])
            ax.grid(True)

            min_y = [np.min([v[0] for v in self._recorded_data['torque']]),
                     np.min([v[1] for v in self._recorded_data['torque']]),
                     np.min([v[2] for v in self._recorded_data['torque']])]

            max_y = [np.max([v[0] for v in self._recorded_data['torque']]),
                     np.max([v[1] for v in self._recorded_data['torque']]),
                     np.max([v[2] for v in self._recorded_data['torque']])]

            ax.set_ylim([np.min(min_y) * 1.1, np.max(max_y) * 1.1])
            ax.set_xlim(np.min(self._time), np.max(self._time))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'disturbance_wrenches.pdf'))
            plt.close(fig)
        except Exception, e:
            self._logger.error('Error while plotting disturbance wrenches, message=' + str(e))
            plt.close(fig)
            del fig
