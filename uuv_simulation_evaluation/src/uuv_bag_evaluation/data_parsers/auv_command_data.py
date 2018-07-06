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


class AUVCommandData(SimulationData):
    LABEL = 'auv_control'

    def __init__(self, bag):
        super(AUVCommandData, self).__init__()

        self._plot_configs = dict(auv_command=dict(
                                    figsize=[12, 5],
                                    linewidth=2,
                                    label_fontsize=30,
                                    xlim=None,
                                    ylim=None,
                                    zlim=None,
                                    tick_labelsize=25,
                                    labelpad=10,
                                    legend=dict(loc='upper right',
                                                fontsize=20)))

        for x in bag.get_type_and_topic_info():
            for k in x:
                if 'uuv_auv_control_allocator/AUVCommand' in x[k][0]:
                    self._topic_name = k
                    self._logger.info('AUV wrench command topic found <%s>' % k)
                    break
            if self._topic_name is not None:
                break

        try:
            self._time = list()
            self._recorded_data['force'] = list()
            self._recorded_data['torque'] = list()
            self._recorded_data['surge_speed'] = list()
            for topic, msg, time in bag.read_messages(self._topic_name):
                    time = msg.header.stamp.to_sec()
                    self._time.append(time)
                    self._recorded_data['force'].append([msg.command.force.x, msg.command.force.y, msg.command.force.z])
                    self._recorded_data['torque'].append([msg.command.torque.x, msg.command.torque.y, msg.command.torque.z])
                    self._recorded_data['surge_speed'].append(float(msg.surge_speed))
            self._logger.info('%s=loaded' % self._topic_name)
        except Exception, e:
            self._logger.error('Error reading AUV command input topic, message=' + str(e))

    def plot(self, output_dir):
        if not os.path.isdir(output_dir):
            self._logger.error('Invalid output directory, dir=' + str(output_dir))
            raise Exception('Invalid output directory')
        fig_tm, ax_tm = plt.subplots(
            3, 1,
            figsize=(self._plot_configs['auv_command']['figsize'][0],
                        3 * self._plot_configs['auv_command']['figsize'][1]))
        try:
            ##############################################################################
            # Plot AUV input command
            ##############################################################################            
            self._logger.info('Plotting the input wrench of the thruster manager')

            min_y = 0
            max_y = 0
            
            ax_tm[0].plot(self._time, [f[0] for f in self._recorded_data['force']],
                          linewidth=self._plot_configs['auv_command']['linewidth'],
                          label=r'$X$')
            min_y = min(min_y, np.min([f[0] for f in self._recorded_data['force']]))
            max_y = max(max_y, np.max([f[0] for f in self._recorded_data['force']]))
            ax_tm[0].plot(self._time, [f[1] for f in self._recorded_data['force']],
                            linewidth=self._plot_configs['auv_command']['linewidth'],
                            label=r'$Y$')
            min_y = min(min_y, np.min([f[1] for f in self._recorded_data['force']]))
            max_y = max(max_y, np.max([f[1] for f in self._recorded_data['force']]))
            ax_tm[0].plot(self._time, [f[2] for f in self._recorded_data['force']],
                            linewidth=self._plot_configs['auv_command']['linewidth'],
                            label=r'$Z$')
            min_y = min(min_y, np.min([f[2] for f in self._recorded_data['force']]))
            max_y = max(max_y, np.max([f[2] for f in self._recorded_data['force']]))

            ax_tm[0].set_xlabel(r'Time [s]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[0].set_ylabel(r'Forces [N]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[0].tick_params(axis='both',
                                    labelsize=self._plot_configs['auv_command']['tick_labelsize'])
            ax_tm[0].grid(True)
            ax_tm[0].set_xlim(np.min(self._time), np.max(self._time))            
            ax_tm[0].set_ylim(min_y, max_y)

            ax_tm[0].legend(fancybox=True, framealpha=1,
                            loc=self._plot_configs['auv_command']['legend']['loc'],
                            fontsize=self._plot_configs['auv_command']['legend']['fontsize'])

            min_y = 0
            max_y = 0

            ax_tm[1].plot(self._time, [x[0] for x in self._recorded_data['torque']],
                            linewidth=self._plot_configs['auv_command']['linewidth'],
                            label=r'$K$')
            min_y = min(min_y, np.min([x[0] for x in self._recorded_data['torque']]))
            max_y = max(max_y, np.max([x[0] for x in self._recorded_data['torque']]))
            ax_tm[1].plot(self._time, [x[1] for x in self._recorded_data['torque']],
                            linewidth=self._plot_configs['auv_command']['linewidth'],
                            label=r'$M$')
            min_y = min(min_y, np.min([x[1] for x in self._recorded_data['torque']]))
            max_y = max(max_y, np.max([x[1] for x in self._recorded_data['torque']]))
            ax_tm[1].plot(self._time, [x[2] for x in self._recorded_data['torque']],
                            linewidth=self._plot_configs['auv_command']['linewidth'],
                            label=r'$N$')     
            min_y = min(min_y, np.min([x[2] for x in self._recorded_data['torque']]))
            max_y = max(max_y, np.max([x[2] for x in self._recorded_data['torque']]))               
            
            ax_tm[1].set_xlabel(r'Time [s]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[1].set_ylabel(r'Torques [Nm]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[1].tick_params(axis='both',
                                 labelsize=self._plot_configs['auv_command']['tick_labelsize'])
            ax_tm[1].grid(True)
            ax_tm[1].set_xlim(np.min(self._time), np.max(self._time))            
            ax_tm[1].set_ylim(min_y, max_y)

            ax_tm[1].legend(fancybox=True, framealpha=1,
                            loc=self._plot_configs['auv_command']['legend']['loc'],
                            fontsize=self._plot_configs['auv_command']['legend']['fontsize'])

            min_y = 0
            max_y = 0

            ax_tm[2].plot(self._time, self._recorded_data['surge_speed'],
                          linewidth=self._plot_configs['auv_command']['linewidth'],
                          label=r'$U$')
            min_y = np.min(self._recorded_data['surge_speed'])
            max_y = np.max(self._recorded_data['surge_speed'])
            
            ax_tm[2].set_xlabel(r'Time [s]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[2].set_ylabel(r'Surge speed [m/s]',
                                fontsize=self._plot_configs['auv_command']['label_fontsize'])
            ax_tm[2].tick_params(axis='both',
                                 labelsize=self._plot_configs['auv_command']['tick_labelsize'])
            ax_tm[2].grid(True)
            ax_tm[2].set_xlim(np.min(self._time), np.max(self._time))            
            ax_tm[2].set_ylim(min_y, max_y)

            ax_tm[2].legend(fancybox=True, framealpha=1,
                            loc=self._plot_configs['auv_command']['legend']['loc'],
                            fontsize=self._plot_configs['auv_command']['legend']['fontsize'])

            fig_tm.tight_layout()
            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'auv_control_input.pdf')
            fig_tm.savefig(filename)
            plt.close(fig_tm)
            del fig_tm
        except Exception, e:
            self._logger.error('Error plotting AUV command input command wrench, message=' + str(e))
            plt.close(fig_tm)
            del fig_tm