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
import matplotlib.pyplot as plt
import os
from simulation_data import SimulationData


class FinsData(SimulationData):
    LABEL = 'fins'

    def __init__(self, bag):
        super(FinsData, self).__init__()

        self._plot_configs = dict(fin_plots=dict(
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
                                        fontsize=20)))
        
        for x in bag.get_type_and_topic_info():
            for k in x:
                if self._prefix is None:
                    if 'fins' in k:
                        rospy.loginfo('Fins topic prefix found <%s>' % k)
                        i = len('fins')
                        i_max = k.find('fins') + i
                        self._prefix = k[:i_max]
                        break
            if self._prefix is not None:
                break
    
        try:
            # Find all fins input topics
            if self._prefix is not None:
                for i in range(16):
                    for topic, msg, time in bag.read_messages('%s/%d/input' % (self._prefix, i)):
                        if i not in self._recorded_data:
                            self._recorded_data[i] = dict(input=dict(time=list(), values=list()))
                        t = msg.header.stamp.to_sec()
                        self._recorded_data[i]['input']['time'].append(t)
                        self._recorded_data[i]['input']['values'].append(float(msg.data))
                    if i in self._recorded_data:
                        self._logger.info('%s/%d/input=loaded' % (self._prefix, i))
        except Exception as e:
            self._logger.error('Error retrieving fin input data from rosbag, message=' + str(e))

        try:
            # Find all fins output topics
            if self._prefix is not None:
                for i in range(16):
                    for topic, msg, time in bag.read_messages('%s/%d/output' % (self._prefix, i)):
                        if 'output' not in self._recorded_data[i]:
                            self._recorded_data[i]['output'] = dict(time=list(), values=list())
                        t = msg.header.stamp.to_sec()
                        self._recorded_data[i]['output']['time'].append(t)
                        self._recorded_data[i]['output']['values'].append(float(msg.data))
                    if i in self._recorded_data:
                        self._logger.info('%s/%d/output=loaded' % (self._prefix, i))
        except Exception as e:
            self._logger.error('Error retrieving fin output data from rosbag, message=' + str(e))

        try:
            # Find all fin wrench topics
            if self._prefix is not None:
                for i in range(16):
                    for topic, msg, time in bag.read_messages('%s/%d/wrench_topic' % (self._prefix, i)):
                        if 'wrench' not in self._recorded_data[i]:
                            self._recorded_data[i]['wrench'] = dict(time=list(), force=list(), torque=list())
                        time = msg.header.stamp.to_sec()
                        self._recorded_data[i]['wrench']['time'].append(time)
                        self._recorded_data[i]['wrench']['force'].append(
                            [msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z])
                        self._recorded_data[i]['wrench']['torque'].append(
                            [msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
                    if i in self._recorded_data:
                        self._logger.info('%s/%d/wrench_topic=loaded' % (self._prefix, i))
        except Exception as e:
            self._logger.error('Error retrieving fin wrench data from rosbag, message=' + str(e))

    @property
    def n_fins(self):
        return len(self._recorded_data.keys())

    def plot(self, output_dir):
        if not os.path.isdir(output_dir):
            rospy.logerr('Invalid output directory, dir=' + str(output_dir))
            raise Exception('Invalid output directory')

        fig_all = plt.figure(figsize=(self._plot_configs['fin_plots']['figsize'][0], 
                                      self._plot_configs['fin_plots']['figsize'][1]))
        try:
            ##############################################################################
            # All fin outputs
            ##############################################################################
            ax_all = fig_all.gca()
            max_y = 0.0
            min_t = 0.0
            max_t = 0.0
            for i in self._recorded_data.keys():
                ax_all.plot(self._recorded_data[i]['output']['time'],
                            self._recorded_data[i]['output']['values'],
                            linewidth=self._plot_configs['fin_plots']['linewidth'],
                            label='%d' % i)

                max_y = np.max([max_y, np.max(np.abs(self._recorded_data[i]['output']['values']))])
                min_t = np.min([min_t, np.min(self._recorded_data[i]['output']['time'])])
                max_t = np.max([max_t, np.max(self._recorded_data[i]['output']['time'])])
            ax_all.set_xlabel('Time [s]',
                              fontsize=self._plot_configs['fin_plots']['label_fontsize'])
            ax_all.set_ylabel(r'Angle [rad]',
                              fontsize=self._plot_configs['fin_plots']['label_fontsize'])
            ax_all.tick_params(axis='both',
                               labelsize=self._plot_configs['fin_plots']['tick_labelsize'])
            ax_all.grid(True)
            ax_all.set_xlim(min_t, max_t)
            ax_all.set_ylim(-max_y - 0.05, max_y + 0.05)

            ax_all.legend(fancybox=True, framealpha=1,
                          loc=self._plot_configs['fin_plots']['legend']['loc'],
                          fontsize=self._plot_configs['fin_plots']['legend']['fontsize'])

            plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
            fig_all.tight_layout()

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'fin_angles_output_all.pdf')
            fig_all.savefig(filename)
            plt.close(fig_all)
            del fig_all
        except Exception as e:
            self._logger.error('Error plotting all fin output angles, message=' + str(e))
            plt.close(fig_all)
            del fig_all

        fig_all = plt.figure(figsize=(self._plot_configs['fin_plots']['figsize'][0], 
                                      self._plot_configs['fin_plots']['figsize'][1]))
        try:
            ##############################################################################
            # All fin inputs
            ##############################################################################            
            ax_all = fig_all.gca()
            max_y = 0.0
            min_t = 0.0
            max_t = 0.0
            for i in self._recorded_data.keys():
                ax_all.plot(self._recorded_data[i]['input']['time'],
                            self._recorded_data[i]['input']['values'],
                            linewidth=self._plot_configs['fin_plots']['linewidth'],
                            label='%d' % i)

                max_y = np.max([max_y, np.max(np.abs(self._recorded_data[i]['input']['values']))])
                min_t = np.min([min_t, np.min(self._recorded_data[i]['input']['time'])])
                max_t = np.max([max_t, np.max(self._recorded_data[i]['input']['time'])])
            ax_all.set_xlabel('Time [s]',
                              fontsize=self._plot_configs['fin_plots']['label_fontsize'])
            ax_all.set_ylabel(r'Angle [rad]',
                              fontsize=self._plot_configs['fin_plots']['label_fontsize'])
            ax_all.tick_params(axis='both',
                               labelsize=self._plot_configs['fin_plots']['tick_labelsize'])
            ax_all.grid(True)
            ax_all.set_xlim(min_t, max_t)
            ax_all.set_ylim(-max_y - 0.05, max_y + 0.05)

            ax_all.legend(fancybox=True, framealpha=1,
                          loc=self._plot_configs['fin_plots']['legend']['loc'],
                          fontsize=self._plot_configs['fin_plots']['legend']['fontsize'])

            plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
            fig_all.tight_layout()

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'fin_angles_input_all.pdf')
            fig_all.savefig(filename)
            plt.close(fig_all)
            del fig_all
        except Exception as e:
            self._logger.error('Error plotting all fin input command angles, message=' + str(e))
            plt.close(fig_all)
            del fig_all