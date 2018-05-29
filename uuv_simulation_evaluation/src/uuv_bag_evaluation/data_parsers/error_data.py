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
import uuv_bag_evaluation.error 
from uuv_trajectory_generator import TrajectoryGenerator, TrajectoryPoint


class ErrorData(SimulationData):
    LABEL = 'error'

    def __init__(self, bag):
        super(ErrorData, self).__init__(message_type='uuv_control_msgs/TrajectoryPoint')

        self._plot_configs = dict(errors=dict(
                                    figsize=[12, 5],
                                    linewidth=2,
                                    label_fontsize=20,
                                    xlim=None,
                                    ylim=None,
                                    zlim=None,
                                    tick_labelsize=18,
                                    labelpad=10,
                                    legend=dict(
                                        loc='upper right',
                                        fontsize=18)),
                                 error_dist=dict(figsize=[12, 5],
                                    linewidth=3,
                                    label_fontsize=30,
                                    xlim=None,
                                    ylim=None,
                                    zlim=None,
                                    tick_labelsize=25,
                                    labelpad=10,
                                    legend=dict(
                                        loc='upper right',
                                        fontsize=18)))

        for x in bag.get_type_and_topic_info():
            for k in x:
                if 'error' in k and self._message_type in x[k][0]:
                    self._topic_name = k
                    self._logger.info('Error topic found <%s>' % k)
                    break
            if self._topic_name is not None:
                break

        try:
            self._recorded_data['error'] = TrajectoryGenerator()
            for topic, msg, time in bag.read_messages(self._topic_name):
                self._recorded_data['error'].add_trajectory_point_from_msg(msg)
            self._logger.info('%s=loaded' % self._topic_name)
        except Exception, e:
            self._logger.error('Error retrieving error data from rosbag, message=' + str(e))
            self._recorded_data['error'] = None

        self._error_set = None

    @property
    def error(self):
        return self._recorded_data['error']
    
    def plot(self, output_dir):
        if not os.path.isdir(output_dir):
            self._logger.error('Invalid output directory, dir=' + str(output_dir))
            raise rospy.ROSException('Invalid output directory')
        
        fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                  2 * self._plot_configs['errors']['figsize'][1]))
        try:
            # Create error set object
            self._error_set = uuv_bag_evaluation.error.ErrorSet.get_instance()

            if self._error_set is None:
                self._logger.info('Error set has not been correctly initialized')
                raise rospy.ROSException('Error set has not been correctly initialized')

            ##################################################################################
            # Plotting position and orientation errors
            ##################################################################################
            output_path = (self._output_dir if output_dir is None else output_dir)

            t = self._error_set.get_time()
            
            ax = fig.add_subplot(211)
            ax.set_title('Position error', fontsize=20)
            ax.plot(t, self._error_set.get_data('x'), 'r', label=r'$X$')
            ax.plot(t, self._error_set.get_data('y'), 'g', label=r'$Y$')
            ax.plot(t, self._error_set.get_data('z'), 'b', label=r'$Z$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [m]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            ax = fig.add_subplot(212)
            ax.set_title('Orientation error', fontsize=20)
            ax.plot(t, self._error_set.get_data('roll'), 'r', label=r'$\phi$')
            ax.plot(t, self._error_set.get_data('pitch'), 'g', label=r'$\theta$')
            ax.plot(t, self._error_set.get_data('yaw'), 'b', label=r'$\psi$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [rad]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'errors_pose.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting pose errors, message=' + str(e))
            plt.close(fig)
            del fig

        fig = plt.figure(figsize=(self._plot_configs['error_dist']['figsize'][0],
                                  2 * self._plot_configs['error_dist']['figsize'][1]))
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            
            ax = fig.add_subplot(211)

            error = np.sqrt([e.dot(e) for e in self._error_set.get_data('position')]) 

            # self.add_disturbance_activation_spans(ax, 0, error.max())

            t = self._error_set.get_time()
            ax.plot(t, error, color='#003300',
                    linewidth=self._plot_configs['error_dist']['linewidth'],
                    label=r'Euc. position error')

            ax.grid(True)
            ax.legend(loc=self._plot_configs['error_dist']['legend']['loc'],
                      fontsize=self._plot_configs['error_dist']['legend']['fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['error_dist']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['error_dist']['label_fontsize'])
            ax.set_ylabel('Euc. position error [m]',
                          fontsize=self._plot_configs['error_dist']['label_fontsize'])
            ax.set_xlim(np.min(t), np.max(t))
            ax.set_ylim(0, np.max(error) * 1.05)

            # Plot heading error
            ax = fig.add_subplot(212)

            error = self._error_set.get_data('yaw')

            # self.add_disturbance_activation_spans(ax, np.min(error), np.max(error))

            t = self._error_set.get_time()
            ax.plot(t, error, color='#003300',
                    linewidth=self._plot_configs['error_dist']['linewidth'],
                    label='Heading error')

            ax.grid(True)
            ax.legend(loc=self._plot_configs['error_dist']['legend']['loc'],
                      fontsize=self._plot_configs['error_dist']['legend']['fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['error_dist']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['error_dist']['label_fontsize'])
            ax.set_ylabel('Heading error [rad]',
                          fontsize=self._plot_configs['error_dist']['label_fontsize'])
            ax.set_xlim(np.min(t), np.max(t))
            ax.set_ylim(np.min(error) * 1.05, np.max(error) * 1.05)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'error_pos_heading.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting position and heading error plots, message=' + str(e))
            plt.close(fig)
            del fig

        fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                  2 * self._plot_configs['errors']['figsize'][1]))
        try:
            ##################################################################################
            # Plotting position and orientation errors
            ##################################################################################
            output_path = (self._output_dir if output_dir is None else output_dir)

            t = self._error_set.get_time()            
            ax = fig.add_subplot(211)
            ax.set_title('Position error', fontsize=20)
            ax.plot(t, self._error_set.get_data('x'), 'r', label=r'$X$')
            ax.plot(t, self._error_set.get_data('y'), 'g', label=r'$Y$')
            ax.plot(t, self._error_set.get_data('z'), 'b', label=r'$Z$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [m]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            ax = fig.add_subplot(212)
            ax.set_title('Orientation error', fontsize=20)
            ax.plot(t, self._error_set.get_data('roll'), 'r', label=r'$\phi$')
            ax.plot(t, self._error_set.get_data('pitch'), 'g', label=r'$\theta$')
            ax.plot(t, self._error_set.get_data('yaw'), 'b', label=r'$\psi$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [rad]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'errors_pose.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting pose errors, message=' + str(e))
            plt.close(fig)
            del fig

        fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                  2 * self._plot_configs['errors']['figsize'][1]))
        try:
            ##################################################################################
            # Plotting velocity errors
            ##################################################################################                        
            ax = fig.add_subplot(211)
            ax.set_title('Linear velocity error', fontsize=20)
            ax.plot(t, [e[0] for e in self._error_set.get_data('linear_velocity')], 'r', label=r'$\dot{X}$')
            ax.plot(t, [e[1] for e in self._error_set.get_data('linear_velocity')], 'g', label=r'$\dot{Y}$')
            ax.plot(t, [e[2] for e in self._error_set.get_data('linear_velocity')], 'b', label=r'$\dot{Z}$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [m/s]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            ax = fig.add_subplot(212)
            ax.set_title('Angular velocity error', fontsize=20)
            ax.plot(t, [e[0] for e in self._error_set.get_data('angular_velocity')], 'r', label=r'$\omega_x$')
            ax.plot(t, [e[1] for e in self._error_set.get_data('angular_velocity')], 'g', label=r'$\omega_y$')
            ax.plot(t, [e[2] for e in self._error_set.get_data('angular_velocity')], 'b', label=r'$\omega_z$')
            ax.legend(fancybox=True, framealpha=0.9, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [rad/s]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'errors_vel.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting velocity errors, message=' + str(e))
            plt.close(fig)
            del fig
        
        fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                  self._plot_configs['errors']['figsize'][1]))
        try:    
            ##################################################################################
            # Plotting quaternion vector errors
            ##################################################################################

            t = self._error_set.get_time()            
            ax = fig.add_subplot(111)
            ax.set_title('Quaternion orientation error', fontsize=20)
            ax.plot(t, [e[0] for e in self._error_set.get_data('quaternion')], 'r', label=r'$\epsilon_x$')
            ax.plot(t, [e[1] for e in self._error_set.get_data('quaternion')], 'g', label=r'$\epsilon_y$')
            ax.plot(t, [e[2] for e in self._error_set.get_data('quaternion')], 'b', label=r'$\epsilon_z$')
            ax.legend(fancybox=True, framealpha=1, loc='upper right', fontsize=18)
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'errors_quat.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting quaternion vector error, message=' + str(e))
            plt.close(fig)
            del fig

        fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                  self._plot_configs['errors']['figsize'][1]))
        try:    
            ##################################################################################
            # Plotting cross-track errors
            ##################################################################################

            t = self._error_set.get_time('desired')
            
            ax = fig.add_subplot(111)
            ax.set_title('Cross-track error', fontsize=20)
            ax.plot(t, self._error_set.get_data('cross_track'), 'r')
            ax.grid(True)
            ax.tick_params(axis='both', labelsize=16)
            ax.set_xlabel('Time [s]', fontsize=18)
            ax.set_ylabel('Error [m]', fontsize=18)
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'errors_cross_track.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting cross-track error, message=' + str(e))
            plt.close(fig)
            del fig