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

import os
import sys
import numpy as np
import yaml
import tf.transformations as trans
from copy import deepcopy
from .recording import Recording
from .error import ErrorSet
from .metrics import KPI
import matplotlib.pyplot as plt
import logging
from mpl_toolkits.mplot3d import Axes3D

try:
    plt.rc('text', usetex=True)
    plt.rc('font', family='sans-serif')
except Exception, e:
    print 'Cannot use Latex configuration with matplotlib, message=', str(e)

class Evaluation(object):
    PLOT_CONFIG = dict(paths=dict(figsize=[12, 10],
                                linewidth=3,
                                label_fontsize=20,
                                xlim=None,
                                ylim=None,
                                zlim=None,
                                tick_labelsize=18,
                                labelpad=10,
                                legend=dict(loc='upper right',
                                            fontsize=18)),
                     trajectories=dict(figsize=[12, 18],
                                       linewidth=2,
                                       label_fontsize=24,
                                       title_fontsize=26,
                                       xlim=None,
                                       ylim=None,
                                       zlim=None,
                                       tick_labelsize=22,
                                       labelpad=10,
                                       legend=dict(loc='upper right',
                                                   fontsize=20)),
                     errors=dict(figsize=[12, 5],
                                 linewidth=2,
                                 label_fontsize=20,
                                 xlim=None,
                                 ylim=None,
                                 zlim=None,
                                 tick_labelsize=18,
                                 labelpad=10,
                                 legend=dict(loc='upper right',
                                             fontsize=18)),
                     thruster_output=dict(figsize=[12, 5],
                                          linewidth=2,
                                          label_fontsize=30,
                                          xlim=None,
                                          ylim=None,
                                          zlim=None,
                                          tick_labelsize=25,
                                          labelpad=10,
                                          legend=dict(loc='upper right',
                                                      fontsize=20)),
                     current=dict(figsize=[12, 5],
                                  linewidth=2,
                                  label_fontsize=30,
                                  xlim=None,
                                  ylim=None,
                                  zlim=None,
                                  tick_labelsize=25,
                                  labelpad=10,
                                  legend=dict(loc='upper right',
                                              fontsize=25)),
                     wrenches=dict(figsize=[12, 5],
                                   linewidth=2,
                                   label_fontsize=30,
                                   xlim=None,
                                   ylim=None,
                                   zlim=None,
                                   tick_labelsize=25,
                                   labelpad=10,
                                   legend=dict(loc='upper right',
                                               fontsize=18)),
                     error_dist=dict(figsize=[12, 5],
                                     linewidth=3,
                                     label_fontsize=30,
                                     xlim=None,
                                     ylim=None,
                                     zlim=None,
                                     tick_labelsize=25,
                                     labelpad=10,
                                     legend=dict(loc='upper right',
                                                 fontsize=18)))

    def __init__(self, filename, output_dir='.', plot_config_file=None, time_offset=0.0):
        # Setting up the log
        self._logger = logging.getLogger('run_evaluation')
        if len(self._logger.handlers) == 0:
            out_hdlr = logging.StreamHandler(sys.stdout)
            out_hdlr.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(module)s | %(message)s'))
            out_hdlr.setLevel(logging.INFO)
            self._logger.addHandler(out_hdlr)
            self._logger.setLevel(logging.INFO)

        self._logger.info('Opening bag: %s' % filename)
        self._bag = Recording(filename)
        if not self._bag.is_init:
            self._logger.error('Recording could not be opened')
            raise Exception('Recording could not be opened')

        # Create error set object
        self._error_set = ErrorSet.get_instance()
        if self._error_set is None:
            self._logger.error('Error set has not been correctly initialized')
            raise Exception('Error set has not been correctly initialized')

        self._error_set.compute_errors()

        # Assigning the output directory for the results
        if not os.path.isdir(output_dir):
            self._logger.error('Invalid output directory, dir=%s' % str(output_dir) )
            raise Exception('Invalid output directory')

        # Simulation time offset to start the computation of each KPI
        if time_offset >= 0.0:
            self._time_offset = time_offset
        else:
            self._logger.error('Invalid time offset, setting time offset to zero')
            self._time_offset = 0.0

        self._logger.info('Time offset for KPI evaluation [s]=' + str(self._time_offset))

        self._output_dir = output_dir
        # Table of configuration parameters (set per default all KPIs)
        self._kpis = list()
        for kpi in KPI.get_all_kpi_tags():
            if KPI.get_kpi_target(kpi) == 'error':
                for error_tag in self._error_set.get_tags():
                    self._kpis.append(dict(func=KPI.get_kpi(kpi, error_tag),
                                           value=0.0))
            else:
                self._kpis.append(dict(func=KPI.get_kpi(kpi),
                                       value=0.0))

        self._cost_fcn_terms = dict()

        # Setting the default plot configuration data
        self._plot_configs = deepcopy(self.PLOT_CONFIG)
        # The plot configurations can be overwritten using an external YAML
        # file, but plot configurations not present in the file will be kept in
        # its default value
        try:
            if plot_config_file is not None:
                if os.path.isfile(plot_config_file):
                    with open(plot_config_file, 'r') as p_file:
                        p_config = yaml.load(p_file)
                    for tag in p_config:
                        if tag not in self._plot_configs:
                            self._logger.info('Ignoring invalid plot configuration tag, tag=%s' % tag)
                            continue
                        for k in p_config[tag]:
                            if k in self._plot_configs[tag]:
                                if type(p_config[tag][k]) == type(self._plot_configs[tag][k]):
                                    if k == 'legend':
                                        if 'loc' not in p_config[tag][k] or 'fontsize' not in p_config[tag][k]:
                                            self._logger.info('Invalid legend configuration data')
                                            continue
                                    self._plot_configs[tag][k] = p_config[tag][k]
                                else:
                                    self._logger.info('Plot configuration in the wrong data type, tag=', tag + '/' + k)
                else:
                    self._logger.warning('Invalid plot configuration file, using default values instead')
        except Exception, e:
            self._logger.error('Error setting plot configurations, using default values instead, msg=' + str(e))
            self._plot_configs = deepcopy(self.PLOT_CONFIG)

        # Calculating the KPIs for this bag
        self.compute_kpis()

    def __del__(self):
      if self._bag is not None:
        del self._bag

    @property
    def error_set(self):
        return self._error_set

    def calc_cost_fcn(self):
        cost = 0.0
        for tag in self._cost_fcn_terms:
            cost += self._cost_fcn_terms[tag] * self.get_kpi(tag)
        return cost

    def save_cost_fcn_config(self, filename):
        try:
            with open(filename, 'w') as cost_file:
                yaml.dump(self._cost_fcn_terms, cost_file,
                          default_flow_style=False)
        except Exception, e:
            self._logger.error('Error exporting cost function configuration, message=' + str(e))

    def load_cost_fcn(self, filename):
        if not os.path.isfile(filename):
            self._logger.error('Invalid filename, file=%s' % filename)
            return False
        with open(filename, 'w') as fcn_file:
            fcn = yaml.load(fcn_file)
        try:
            for item in fcn:
                self.add_cost_fcn_term(item, fcn[item])
                self._logger.info('Cost function term (tag, weight): (%s, %.4f)' % (item, fcn[item]))
        except Exception, e:
            self._logger.error('Error loading cost function configuration')
            self._logger.error(e)
            return False
        return True

    def add_cost_fcn_term(self, kpi, weight):
        if weight <= 0:
            self._logger.error('Weight must be a positive value')
            return False
        if not self.has_kpi(kpi):
            self._logger.error('KPI tag is invalid, tag=%s' % kpi)
            return False
        if kpi in self._cost_fcn_terms:
            self._logger.error('KPI already added to the cost function, tag=%s' % kpi)
            return False

        self._cost_fcn_terms[kpi] = weight
        return True

    def has_kpi(self, tag):
        for kpi in self._kpis:
            if tag == kpi.full_tag:
                return True
        return False

    def set_kpis_from_file(self, filename):
        assert os.path.isfile(filename), 'Invalid evaluation configuration file'
        assert '.yaml' in filename or '.yml' in filename, 'Configuration file should be a YAML file'

        with open(filename, 'r') as config_file:
            config = yaml.load(config_file)

        self.set_kpis(config)

    def set_kpis(self, config):
        assert type(config) == list, 'Invalid configuration structure for KPIs'
        kpi_tags = KPI.get_all_kpi_tags()

        self._kpis = list()
        for item in config:
            if item['func'] not in kpi_tags:
                self._logger.error('Invalid KPI tag, value=' + item)
            else:
                self._logger.info('KPI created=' + item['func'])
                if 'args' in item:
                    self._kpis.append(dict(func=KPI.get_kpi(item['func'], item['args']),
                                           value=0.0))
                    self._logger.info('\tArguments: ' + str(item['args']))
                else:
                    self._kpis.append(dict(func=KPI.get_kpi(item['func']),
                                           value=0.0))

    def get_kpis(self):
        kpis = dict()
        for kpi in self._kpis:
            item = kpi['func']
            try:
                kpis[item.full_tag] = float(item.kpi_value)
            except:
                kpis[item.full_tag] = -1000.0
        return kpis

    def get_kpi(self, tag):
        for kpi in self._kpis:
            if kpi['func'].full_tag == tag:
                return kpi['value']
        return None

    def get_error_time(self):
        return self._error_set.get_time()

    def get_error_from_data(self, tag):
        return self._error_set.get_data(tag)

    def get_error_set_data(self, tag):
        return self._error_set.get_data(tag)

    def get_current_velocity(self):
        return self._bag.get_current_vel()

    def get_wrench_dist(self):
        return self._bag.get_wrench_dist()

    def compute_kpis(self):
        if len(self._kpis):
            for i in range(len(self._kpis)):
                self._kpis[i]['value'] = self._kpis[i]['func'].compute()

    def print_kpis(self):
        for item in self._kpis:
            print item['func'].full_tag, '= ', item['value']

    def get_trajectory_coord(self, tag):
        return self._bag.get_trajectory_coord(tag)

    def export_to_txt(self, tag, output_dir):
        pass

    def plot_paths(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            fig = plt.figure(figsize=(self._plot_configs['paths']['figsize'][0],
                                      self._plot_configs['paths']['figsize'][1]))
            ax = fig.gca(projection='3d')

            ax.plot([e.p[0] for e in self._bag.desired.points],
                    [e.p[1] for e in self._bag.desired.points],
                    [e.p[2] for e in self._bag.desired.points],
                    'b--', label='Reference path',
                    linewidth=self._plot_configs['paths']['linewidth'])

            ax.plot([e.p[0] for e in self._bag.actual.points],
                    [e.p[1] for e in self._bag.actual.points],
                    [e.p[2] for e in self._bag.actual.points],
                    'g', label='Actual path',
                    linewidth=self._plot_configs['paths']['linewidth'])

            ax.plot([self._bag.actual.points[0].p[0]],
                    [self._bag.actual.points[0].p[1]],
                    [self._bag.actual.points[0].p[2]],
                    'ro', label='Starting position',
                    linewidth=self._plot_configs['paths']['linewidth'])

            # Calculating the boundaries of the paths
            min_x = np.min([np.min([e.p[0] for e in self._bag.desired.points]),
                            np.min([e.p[0] for e in self._bag.actual.points])])

            max_x = np.max([np.max([e.p[0] for e in self._bag.desired.points]),
                            np.max([e.p[0] for e in self._bag.actual.points])])

            min_y = np.min([np.min([e.p[1] for e in self._bag.desired.points]),
                            np.min([e.p[1] for e in self._bag.actual.points])])

            max_y = np.max([np.max([e.p[1] for e in self._bag.desired.points]),
                            np.max([e.p[1] for e in self._bag.actual.points])])

            min_z = np.min([np.min([e.p[2] for e in self._bag.desired.points]),
                            np.min([e.p[2] for e in self._bag.actual.points])])

            max_z = np.max([np.max([e.p[2] for e in self._bag.desired.points]),
                            np.max([e.p[2] for e in self._bag.actual.points])])

            ax.set_xlabel('X [m]',
                          fontsize=self._plot_configs['paths']['label_fontsize'])
            ax.set_ylabel('Y [m]',
                          fontsize=self._plot_configs['paths']['label_fontsize'])
            ax.set_zlabel('Z [m]',
                          fontsize=self._plot_configs['paths']['label_fontsize'])

            if self._plot_configs['paths']['xlim'] is not None:
                ax.set_xlim(self._plot_configs['paths']['xlim'][0],
                            self._plot_configs['paths']['xlim'][1])
            else:
                ax.set_xlim(min_x - 1, max_x + 1)

            if self._plot_configs['paths']['ylim'] is not None:
                ax.set_ylim(self._plot_configs['paths']['ylim'][0],
                            self._plot_configs['paths']['ylim'][1])
            else:
                ax.set_ylim(min_y - 1, max_y + 1)

            if self._plot_configs['paths']['zlim'] is not None:
                ax.set_zlim(self._plot_configs['paths']['zlim'][0],
                            self._plot_configs['paths']['zlim'][1])
            else:
                ax.set_zlim(min_z - 1, max_z + 1)

            ax.tick_params(axis='x',
                           labelsize=self._plot_configs['paths']['tick_labelsize'])
            ax.tick_params(axis='y',
                           labelsize=self._plot_configs['paths']['tick_labelsize'])
            ax.tick_params(axis='z',
                           labelsize=self._plot_configs['paths']['tick_labelsize'])

            ax.xaxis.labelpad = self._plot_configs['paths']['labelpad']
            ax.yaxis.labelpad = self._plot_configs['paths']['labelpad']
            ax.zaxis.labelpad = self._plot_configs['paths']['labelpad']

            ax.legend(loc=self._plot_configs['paths']['legend']['loc'],
                      fancybox=True,
                      framealpha=0.9,
                      fontsize=self._plot_configs['paths']['legend']['fontsize'])
            ax.grid(True)
            plt.tight_layout()

            # Invert axis if the pose of the vehicle is represented wrt NED
            # inertial reference frame
            if min_z >= 0 and max_z >= 0:
                plt.gca().invert_zaxis()

            ax.view_init(elev=15, azim=30)

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'paths.pdf')
            plt.savefig(filename)
            plt.close(fig)
        except Exception, e:
            self._logger.error('Error while plotting 3D path plot')
            self._logger.error(e)

    def plot_trajectories(self, output_dir):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            
            fig = plt.figure(figsize=(self._plot_configs['trajectories']['figsize'][0],
                                      self._plot_configs['trajectories']['figsize'][1]))
            ax = fig.add_subplot(211)

            min_value = np.min([np.min([e.pos[0] for e in self._bag.actual.points]),
                                np.min([e.pos[0] for e in self._bag.desired.points]),
                                np.min([e.pos[1] for e in self._bag.actual.points]),
                                np.min([e.pos[1] for e in self._bag.desired.points]),
                                np.min([e.pos[2] for e in self._bag.actual.points]),
                                np.min([e.pos[2] for e in self._bag.desired.points])])

            max_value = np.max([np.max([e.pos[0] for e in self._bag.actual.points]),
                                np.max([e.pos[0] for e in self._bag.desired.points]),
                                np.max([e.pos[1] for e in self._bag.actual.points]),
                                np.max([e.pos[1] for e in self._bag.desired.points]),
                                np.max([e.pos[2] for e in self._bag.actual.points]),
                                np.max([e.pos[2] for e in self._bag.desired.points])])

            self.add_disturbance_activation_spans(ax, min_value, max_value)

            ax.set_title('Position',
                         fontsize=self._plot_configs['trajectories']['title_fontsize'])
            ax.plot(self._bag.desired.time, 
                    [e.pos[0] for e in self._bag.desired.points], 'r--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$X_d$')
            ax.plot(self._bag.desired.time, 
                    [e.pos[1] for e in self._bag.desired.points], 'g--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$Y_d$')
            ax.plot(self._bag.desired.time, 
                    [e.pos[2] for e in self._bag.desired.points], 'b--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$Z_d$')

            ax.plot(self._bag.actual.time, 
                    [e.pos[0] for e in self._bag.actual.points], 'r',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$X$')
            ax.plot(self._bag.actual.time, 
                    [e.pos[1] for e in self._bag.actual.points], 'g',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$Y$')
            ax.plot(self._bag.actual.time, 
                    [e.pos[2] for e in self._bag.actual.points], 'b',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$Z$')

            ax.legend(fancybox=True,
                      framealpha=0.7,
                      loc=self._plot_configs['trajectories']['legend']['loc'],
                      fontsize=self._plot_configs['trajectories']['legend']['fontsize'])
            ax.grid(True)
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['trajectories']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_ylabel('Position [m]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_xlim(np.min(self._bag.desired.time), np.max(self._bag.desired.time))
            ax.set_ylim(1.05 * min_value, 1.05 * max_value)

            ax = fig.add_subplot(212)

            min_value = np.min([np.min([e.rot[0] for e in self._bag.actual.points]),
                                np.min([e.rot[0] for e in self._bag.desired.points]),
                                np.min([e.rot[1] for e in self._bag.actual.points]),
                                np.min([e.rot[1] for e in self._bag.desired.points]),
                                np.min([e.rot[2] for e in self._bag.actual.points]),
                                np.min([e.rot[2] for e in self._bag.desired.points])])

            max_value = np.max([np.max([e.rot[0] for e in self._bag.actual.points]),
                                np.max([e.rot[0] for e in self._bag.desired.points]),
                                np.max([e.rot[1] for e in self._bag.actual.points]),
                                np.max([e.rot[1] for e in self._bag.desired.points]),
                                np.max([e.rot[2] for e in self._bag.actual.points]),
                                np.max([e.rot[2] for e in self._bag.desired.points])])

            self.add_disturbance_activation_spans(ax, min_value, max_value)

            ax.set_title('Orientation', fontsize=self._plot_configs['trajectories']['title_fontsize'])
            ax.plot(self._bag.desired.time, 
                    [e.rot[0] for e in self._bag.desired.points], 'r--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\phi_d$')
            ax.plot(self._bag.desired.time, 
                    [e.rot[1] for e in self._bag.desired.points], 'g--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\theta_d$')
            ax.plot(self._bag.desired.time, 
                    [e.rot[2] for e in self._bag.desired.points], 'b--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\psi_d$')

            ax.plot(self._bag.actual.time, 
                    [e.rot[0] for e in self._bag.actual.points], 'r',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\phi$')
            ax.plot(self._bag.actual.time, 
                    [e.rot[1] for e in self._bag.actual.points], 'g',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\theta$')
            ax.plot(self._bag.actual.time, 
                    [e.rot[2] for e in self._bag.actual.points], 'b',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\psi$')

            ax.legend(fancybox=True, framealpha=0.7,
                      loc=self._plot_configs['trajectories']['legend']['loc'],
                      fontsize=self._plot_configs['trajectories']['legend']['fontsize'])
            ax.grid(True)
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['trajectories']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_ylabel('Angles [rad]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_xlim(np.min(self._bag.desired.time), np.max(self._bag.desired.time))
            ax.set_ylim(1.05 * min_value, 1.05 * max_value)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'trajectories_pose.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error plotting output pose vectors, error=' + str(e))

        try:
            ###################################################################
            # Plot quaternion trajectories
            ###################################################################
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)

            ax.set_title('Quaternion trajectories',
                         fontsize=self._plot_configs['trajectories']['title_fontsize'])
            ax.plot(self._bag.desired.time, 
                    [e.rotq[0] for e in self._bag.desired.points], 'r--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{x_d}$')
            ax.plot(self._bag.desired.time, 
                    [e.rotq[1] for e in self._bag.desired.points], 'g--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{y_d}$')
            ax.plot(self._bag.desired.time, 
                    [e.rotq[2] for e in self._bag.desired.points], 'b--',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{z_d}$')

            ax.plot(self._bag.actual.time, 
                    [e.rotq[0] for e in self._bag.actual.points], 'r',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{x}$')
            ax.plot(self._bag.actual.time, 
                    [e.rotq[1] for e in self._bag.actual.points], 'g',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{y}$')
            ax.plot(self._bag.actual.time, 
                    [e.rotq[2] for e in self._bag.actual.points], 'b',
                    linewidth=self._plot_configs['trajectories']['linewidth'],
                    label=r'$\epsilon_{z}$')

            ax.legend(fancybox=True, framealpha=0.7,
                      loc=self._plot_configs['trajectories']['legend']['loc'],
                      fontsize=self._plot_configs['trajectories']['legend']['fontsize'])
            ax.grid(True)

            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['trajectories']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_ylabel('Euler parameters',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])

            min_value = np.min([np.min([e.rotq[0] for e in self._bag.actual.points]),
                                np.min([e.rotq[0] for e in self._bag.desired.points]),
                                np.min([e.rotq[1] for e in self._bag.actual.points]),
                                np.min([e.rotq[1] for e in self._bag.desired.points]),
                                np.min([e.rotq[2] for e in self._bag.actual.points]),
                                np.min([e.rotq[2] for e in self._bag.desired.points])])

            max_value = np.max([np.max([e.rotq[0] for e in self._bag.actual.points]),
                                np.max([e.rotq[0] for e in self._bag.desired.points]),
                                np.max([e.rotq[1] for e in self._bag.actual.points]),
                                np.max([e.rotq[1] for e in self._bag.desired.points]),
                                np.max([e.rotq[2] for e in self._bag.actual.points]),
                                np.max([e.rotq[2] for e in self._bag.desired.points])])

            ax.set_xlim(np.min(self._bag.desired.time), np.max(self._bag.desired.time))
            ax.set_ylim(1.05 * min_value, 1.05 * max_value)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'trajectories_quat.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error plotting output quaternion vectors, error=' + str(e))

        try:
            ###################################################################
            # Plot velocities
            ###################################################################

            fig = plt.figure(figsize=(self._plot_configs['trajectories']['figsize'][0],
                                      self._plot_configs['trajectories']['figsize'][1]))
            ax = fig.add_subplot(211)
            ax.set_title('Linear velocity', fontsize=20)
            ax.plot(self._bag.desired.time, 
                    [e.vel[0] for e in self._bag.desired.points], 'r--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{X}_d$')
            ax.plot(self._bag.desired.time, 
                    [e.vel[1] for e in self._bag.desired.points], 'g--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{Y}_d$')
            ax.plot(self._bag.desired.time, 
                    [e.vel[2] for e in self._bag.desired.points], 'b--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{Z}_d$')

            ax.plot(self._bag.actual.time, 
                    [e.vel[0] for e in self._bag.actual.points], 'r',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{X}$')
            ax.plot(self._bag.actual.time, 
                    [e.vel[1] for e in self._bag.actual.points], 'g',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{Y}$')
            ax.plot(self._bag.actual.time, 
                    [e.vel[2] for e in self._bag.actual.points], 'b',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\dot{Z}$')
            ax.legend(fancybox=True, framealpha=0.9,
                      loc=self._plot_configs['trajectories']['legend']['loc'],
                      fontsize=self._plot_configs['trajectories']['legend']['fontsize'])
            ax.grid(True)
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['trajectories']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_ylabel('Velocity [m/s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_xlim(np.min(self._bag.desired.time), np.max(self._bag.desired.time))

            ax = fig.add_subplot(212)
            ax.set_title('Angular velocity', fontsize=self._plot_configs['trajectories']['title_fontsize'])
            ax.plot(self._bag.desired.time, 
                    [e.vel[3] for e in self._bag.desired.points], 'r--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_{x_d}$')
            ax.plot(self._bag.desired.time, 
                    [e.vel[4] for e in self._bag.desired.points], 'g--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_{y_d}$')
            ax.plot(self._bag.desired.time, 
                    [e.vel[5] for e in self._bag.desired.points], 'b--',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_{z_d}$')

            ax.plot(self._bag.actual.time, 
                    [e.vel[3] for e in self._bag.actual.points], 'r',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_x$')
            ax.plot(self._bag.actual.time, 
                    [e.vel[4] for e in self._bag.actual.points], 'g',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_y$')
            ax.plot(self._bag.actual.time, 
                    [e.vel[5] for e in self._bag.actual.points], 'b',
                    linewidth=self._plot_configs['trajectories']['linewidth'], label=r'$\omega_z$')
            ax.legend(fancybox=True, framealpha=0.9,
                      loc=self._plot_configs['trajectories']['legend']['loc'],
                      fontsize=self._plot_configs['trajectories']['legend']['fontsize'])
            ax.grid(True)
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['trajectories']['tick_labelsize'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_ylabel('Angular velocity [rad/s]',
                          fontsize=self._plot_configs['trajectories']['label_fontsize'])
            ax.set_xlim(np.min(self._bag.desired.time), np.max(self._bag.desired.time))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'trajectories_vel.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error plotting output velocities, error=' + str(e))

    def plot_thruster_output(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            ##############################################################################
            # Plot individual thruster outputs
            ##############################################################################
            fig, ax = plt.subplots(self._bag.n_thrusters, 1,
                                   figsize=(self._plot_configs['thruster_output']['figsize'][0],
                                            self._bag.n_thrusters * self._plot_configs['thruster_output']['figsize'][1]))

            max_y = 0.0
            for i in range(self._bag.n_thrusters):
                t, values = self._bag.get_thruster_data(i)

                # Find largest absolute thrust force value
                max_y = np.max([max_y, np.max(np.abs(values))])

                ax[i].plot(t,
                        values,
                        linewidth=self._plot_configs['thruster_output']['linewidth'],
                        label='%d' % i)

                ax[i].set_xlabel('Time [s]',
                                 fontsize=self._plot_configs['thruster_output']['label_fontsize'])
                ax[i].set_ylabel(r'$\tau_%d$ [N]' % i,
                                 fontsize=self._plot_configs['thruster_output']['label_fontsize'])
                ax[i].tick_params(axis='both',
                                  labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
                ax[i].grid(True)
                ax[i].set_xlim(np.min(t), np.max(t))

            for i in range(self._bag.n_thrusters):
                ax[i].set_ylim(-max_y, max_y)

            fig.tight_layout()
            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thrusts.pdf')
            fig.savefig(filename)
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error plotting individual thruster outputs, message=' + str(e))

        try:
            ##############################################################################
            # All thrust outputs
            ##############################################################################
            fig_all = plt.figure(figsize=(self._plot_configs['thruster_output']['figsize'][0], 
                                          self._plot_configs['thruster_output']['figsize'][1]))

            ax_all = fig_all.gca()

            for i in range(self._bag.n_thrusters):
                t, values = self._bag.get_thruster_data(i)

                ax_all.plot(t,
                            values,
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label='%d' % i)

            ax_all.set_xlabel('Time [s]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_all.set_ylabel(r'Thrust output [N]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_all.tick_params(axis='both',
                               labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
            ax_all.grid(True)
            ax_all.set_xlim(np.min(t), np.max(t))
            ax_all.set_ylim(-max_y, max_y)

            ax_all.legend(fancybox=True, framealpha=1,
                          loc=self._plot_configs['thruster_output']['legend']['loc'],
                          fontsize=self._plot_configs['thruster_output']['legend']['fontsize'])

            plt.gcf().subplots_adjust(left=0.15, bottom=0.15)
            fig_all.tight_layout()

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thrusts_all.pdf')
            fig_all.savefig(filename)
            plt.close(fig_all)
            del fig_all
        except Exception, e:
            self._logger.error('Error plotting all thruster output, message=' + str(e))

        try:
            ##############################################################################
            # Average thruster output
            ##############################################################################
            fig_avg = plt.figure(figsize=(self._plot_configs['thruster_output']['figsize'][0], 
                                          self._plot_configs['thruster_output']['figsize'][1]))
            ax_avg = fig_avg.gca()

            t0, values = self._bag.get_thruster_data(0)
            t0 = np.array(t0)
            thrust_sum = np.zeros(t0.shape)
            thrust_max = np.zeros(t0.shape)
            self._logger.info('Computing sum and maximum element-wise values for the thrust forces')
            for i in range(self._bag.n_thrusters):
                t, values = self._bag.get_thruster_data(i)
                thrust_sum += np.interp(t0, t, np.abs(values))
                thrust_max = np.maximum(thrust_max, np.interp(t0, t, np.abs(values)))

            thrust_sum /= self._bag.n_thrusters
            ax_avg.plot(t0,
                        thrust_sum,
                        linewidth=self._plot_configs['thruster_output']['linewidth'],
                        label=r'$%d$' % i)

            ax_avg.set_xlabel('Time [s]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_avg.set_ylabel(r'$\frac{1}{N} \sum_{i=1}^{N} \tau_i$ [N]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_avg.tick_params(axis='both',
                               labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
            ax_avg.grid(True)
            ax_avg.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.gcf().subplots_adjust(left=0.15, bottom=0.15)

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thrusts_avg.pdf')
            fig_avg.savefig(filename)
            plt.close(fig_avg)
            del fig_avg
        except Exception, e:
            self._logger.error('Error plotting average thrust force output, message=' + str(e))

        try:
            ##############################################################################
            # Maximum thruster output for each time step
            ##############################################################################
            fig_max = plt.figure(figsize=(self._plot_configs['thruster_output']['figsize'][0],
                                          self._plot_configs['thruster_output']['figsize'][1]))
            ax_max = fig_max.gca()
            self._logger.info('Plotting maximum element-wise values for the thrust forces')
            ax_max.plot(t0,
                        thrust_max,
                        linewidth=self._plot_configs['thruster_output']['linewidth'],
                        label=r'$%d$' % i)

            ax_max.set_xlabel('Time [s]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_max.set_ylabel(r'max $| \tau_i |$ [N]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_max.tick_params(axis='both',
                               labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
            ax_max.grid(True)
            ax_max.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.gcf().subplots_adjust(left=0.15, bottom=0.15)

            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thrusts_max.pdf')
            fig_max.savefig(filename)
            plt.close(fig_max)
            del fig_max
        except Exception, e:
            self._logger.error('Error plotting maximum thruster output, message=' + str(e))

        try:
            ##############################################################################
            # Plot thruster command input data
            ##############################################################################
            fig_in = plt.figure(figsize=(self._plot_configs['thruster_output']['figsize'][0],
                                         self._plot_configs['thruster_output']['figsize'][1]))
            ax_min = fig_in.gca()
            self._logger.info('Plotting the input command signals for each thruster unit')

            min_t = 0
            max_t = 0

            min_y = 0
            max_y = 0
            for i in range(self._bag.n_thrusters):
                t, values = self._bag.get_thruster_input_data(i)
                ax_min.plot(t,
                        values,
                        linewidth=self._plot_configs['thruster_output']['linewidth'],
                        label='%d' % i)
                max_t = max(max_t, np.max(t))
                min_y = min(min_y, np.min(values))
                max_y = max(max_y, np.max(values))

            ax_min.set_xlabel(r'Time [s]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_min.set_ylabel(r'Thrust input [rad/s]',
                              fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_min.tick_params(axis='both',
                               labelsize=self._plot_configs['thruster_output']['tick_labelsize'])

            ax_min.legend(fancybox=True, framealpha=1,
                          loc=self._plot_configs['thruster_output']['legend']['loc'],
                          fontsize=self._plot_configs['thruster_output']['legend']['fontsize'])

            ax_min.grid(True)
            ax_min.set_xlim(min_t, max_t)            
            ax_min.set_ylim(min_y, max_y)

            fig_in.tight_layout()
            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thruster_input.pdf')
            fig_in.savefig(filename)
            plt.close(fig_in)
            del fig_in
        except Exception, e:
            self._logger.error('Error plotting thruster input command, message=' + str(e))

        try:
            ##############################################################################
            # Plot thruster manager input data
            ##############################################################################
            fig_tm, ax_tm = plt.subplots(
                2, 1,
                figsize=(self._plot_configs['thruster_output']['figsize'][0],
                         2 * self._plot_configs['thruster_output']['figsize'][1]))
            self._logger.info('Plotting the input wrench of the thruster manager')

            min_y = 0
            max_y = 0
            t, force, torque = self._bag.get_thruster_manager_input()            
            ax_tm[0].plot(t, [f[0] for f in force],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$X$')
            min_y = min(min_y, np.min([f[0] for f in force]))
            max_y = max(max_y, np.max([f[0] for f in force]))
            ax_tm[0].plot(t, [f[1] for f in force],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$Y$')
            min_y = min(min_y, np.min([f[1] for f in force]))
            max_y = max(max_y, np.max([f[1] for f in force]))
            ax_tm[0].plot(t, [f[2] for f in force],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$Z$')
            min_y = min(min_y, np.min([f[2] for f in force]))
            max_y = max(max_y, np.max([f[2] for f in force]))

            ax_tm[0].set_xlabel(r'Time [s]',
                                fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_tm[0].set_ylabel(r'Forces [N]',
                                fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_tm[0].tick_params(axis='both',
                                    labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
            ax_tm[0].grid(True)
            ax_tm[0].set_xlim(np.min(t), np.max(t))            
            ax_tm[0].set_ylim(min_y, max_y)

            ax_tm[0].legend(fancybox=True, framealpha=1,
                            loc=self._plot_configs['thruster_output']['legend']['loc'],
                            fontsize=self._plot_configs['thruster_output']['legend']['fontsize'])

            min_y = 0
            max_y = 0

            ax_tm[1].plot(t, [x[0] for x in torque],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$K$')
            min_y = min(min_y, np.min([x[0] for x in torque]))
            max_y = max(max_y, np.max([x[0] for x in torque]))
            ax_tm[1].plot(t, [x[1] for x in torque],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$M$')
            min_y = min(min_y, np.min([x[1] for x in torque]))
            max_y = max(max_y, np.max([x[1] for x in torque]))
            ax_tm[1].plot(t, [x[2] for x in torque],
                            linewidth=self._plot_configs['thruster_output']['linewidth'],
                            label=r'$N$')     
            min_y = min(min_y, np.min([x[2] for x in torque]))
            max_y = max(max_y, np.max([x[2] for x in torque]))               
            
            ax_tm[1].set_xlabel(r'Time [s]',
                                fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_tm[1].set_ylabel(r'Torques [Nm]',
                                fontsize=self._plot_configs['thruster_output']['label_fontsize'])
            ax_tm[1].tick_params(axis='both',
                                    labelsize=self._plot_configs['thruster_output']['tick_labelsize'])
            ax_tm[1].grid(True)
            ax_tm[1].set_xlim(np.min(t), np.max(t))            
            ax_tm[1].set_ylim(min_y, max_y)

            ax_tm[1].legend(fancybox=True, framealpha=1,
                            loc=self._plot_configs['thruster_output']['legend']['loc'],
                            fontsize=self._plot_configs['thruster_output']['legend']['fontsize'])

            fig_tm.tight_layout()
            output_path = (self._output_dir if output_dir is None else output_dir)
            filename = os.path.join(output_path, 'thruster_manager_input.pdf')
            fig_tm.savefig(filename)
            plt.close(fig_tm)
            del fig_tm
        except Exception, e:
            self._logger.error('Error plotting thruster manager input command wrench, message=' + str(e))

    def add_disturbance_activation_spans(self, ax, min_value, max_value):
        try:
            t, vel = self._bag.get_current_vel()
            if len(t) > 0:
                v = np.array([np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in vel])
                if v.max() > 0:
                    v[v > 0] = 1.05
                    ax.fill_between(t, v * min_value, v * max_value, facecolor='blue', alpha=0.2,
                                    label='Current disturbance activated')

            t, force, torque = self._bag.get_wrench_dist()

            if len(t) > 0:
                f = np.array([np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in force])
                tau = np.array([np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) for v in torque])

                if f.max() > 0:
                    f[f > 0] = 1.05
                    ax.fill_between(t, f * min_value, f * max_value, facecolor='red',
                                    alpha=0.2,
                                    label='Force disturbance activated')

                if tau.max() > 0:
                    tau[tau > 0] = 1.05
                    ax.fill_between(t, tau * min_value, tau * max_value, facecolor='green',
                                    alpha=0.2,
                                    label='Torque disturbance activated')
        except Exception, e:
            self._logger.error('Error while adding disturbance activation time spans, message=' + str(e))

    def plot_error_dist(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)

            fig = plt.figure(figsize=(self._plot_configs['error_dist']['figsize'][0],
                                      2 * self._plot_configs['error_dist']['figsize'][1]))
            ax = fig.add_subplot(211)

            error = KPI.get_error(self._error_set.get_data('position'))

            self.add_disturbance_activation_spans(ax, 0, error.max())

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

            self.add_disturbance_activation_spans(ax, np.min(error), np.max(error))

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

    def plot_errors(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            ##################################################################################
            # Plotting position and orientation errors
            ##################################################################################
            output_path = (self._output_dir if output_dir is None else output_dir)

            t = self._error_set.get_time()
            fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                      2 * self._plot_configs['errors']['figsize'][1]))
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

        try:
            ##################################################################################
            # Plotting velocity errors
            ##################################################################################            
            fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                      2 * self._plot_configs['errors']['figsize'][1]))
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

        try:    
            ##################################################################################
            # Plotting quaternion vector errors
            ##################################################################################

            if self._bag.errors is not None:
                t = self._bag.errors.time
                fig = plt.figure(figsize=(self._plot_configs['errors']['figsize'][0],
                                          self._plot_configs['errors']['figsize'][1]))
                ax = fig.add_subplot(111)
                ax.set_title('Quaternion orientation error', fontsize=20)
                ax.plot(t, [e.rotq[0] for e in self._bag.errors.points], 'r', label=r'$\epsilon_x$')
                ax.plot(t, [e.rotq[1] for e in self._bag.errors.points], 'g', label=r'$\epsilon_y$')
                ax.plot(t, [e.rotq[2] for e in self._bag.errors.points], 'b', label=r'$\epsilon_z$')
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

    def plot_current(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            t, vel = self._bag.get_current_vel()
            if len(t) == 0:
                self._logger.info('No current velocity information')
                return
            fig = plt.figure(figsize=(self._plot_configs['current']['figsize'][0],
                                      self._plot_configs['current']['figsize'][1]))
            ax = fig.add_subplot(111)

            ax.plot(t, [v[0] for v in vel], 'r', label=r'$u_C$',
                    linewidth=self._plot_configs['current']['linewidth'])
            ax.plot(t, [v[1] for v in vel], 'g', label=r'$v_C$',
                    linewidth=self._plot_configs['current']['linewidth'])
            ax.plot(t, [v[2] for v in vel], 'b', label=r'$w_C$',
                    linewidth=self._plot_configs['current']['linewidth'])
            ax.set_xlabel('Time [s]',
                          fontsize=self._plot_configs['current']['label_fontsize'])
            ax.set_ylabel('Velocity [m/s]',
                          fontsize=self._plot_configs['current']['label_fontsize'])
            ax.tick_params(axis='both',
                           labelsize=self._plot_configs['current']['tick_labelsize'])

            min_y = [np.min([v[0] for v in vel]),
                     np.min([v[1] for v in vel]),
                     np.min([v[2] for v in vel])]

            max_y = [np.max([v[0] for v in vel]),
                     np.max([v[1] for v in vel]),
                     np.max([v[2] for v in vel])]

            lim_y = max(np.abs(np.min(min_y)), np.abs(np.max(max_y)))
            ax.set_ylim([np.min(min_y) - 0.1, np.max(max_y) + 0.1])
            # ax.set_yticks(np.linspace(np.floor(np.min(min_y)), np.floor(np.max(max_y)), 0.2))

            ax.legend(fancybox=True, framealpha=0.9,
                      loc=self._plot_configs['current']['legend']['loc'],
                      fontsize=self._plot_configs['current']['legend']['fontsize'])
            ax.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'current_velocity.pdf'))
            plt.close(fig)
            del fig
        except Exception, e:
            self._logger.error('Error while plotting disturbance wrenches, message=' + str(e))

    def plot_wrenches(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)
            t, force, torque = self._bag.get_wrench_dist()

            fig = plt.figure(figsize=(self._plot_configs['wrenches']['figsize'][0],
                                      self._plot_configs['wrenches']['figsize'][1]))
            ax = fig.add_subplot(211)
            ax.plot(t, [f[0] for f in force], 'r', label=r'$F_X$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(t, [f[1] for f in force], 'g', label=r'$F_Y$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(t, [f[2] for f in force], 'b', label=r'$F_Z$',
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

            min_y = [np.min([v[0] for v in force]),
                     np.min([v[1] for v in force]),
                     np.min([v[2] for v in force])]

            max_y = [np.max([v[0] for v in force]),
                     np.max([v[1] for v in force]),
                     np.max([v[2] for v in force])]

            ax.set_ylim([np.min(min_y) * 1.1, np.max(max_y) * 1.1])
            ax.set_xlim(np.min(t), np.max(t))

            ax = fig.add_subplot(212)
            ax.plot(t, [f[0] for f in torque], 'r', label=r'$\tau_X$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(t, [f[1] for f in torque], 'g', label=r'$\tau_Y$',
                    linewidth=self._plot_configs['wrenches']['linewidth'])
            ax.plot(t, [f[2] for f in torque], 'b', label=r'$\tau_Z$',
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

            min_y = [np.min([v[0] for v in torque]),
                     np.min([v[1] for v in torque]),
                     np.min([v[2] for v in torque])]

            max_y = [np.max([v[0] for v in torque]),
                     np.max([v[1] for v in torque]),
                     np.max([v[2] for v in torque])]

            ax.set_ylim([np.min(min_y) * 1.1, np.max(max_y) * 1.1])
            ax.set_xlim(np.min(t), np.max(t))

            plt.tight_layout()
            plt.savefig(os.path.join(output_path, 'disturbance_wrenches.pdf'))
            plt.close(fig)
        except Exception, e:
            self._logger.error('Error while plotting disturbance wrenches, message=' + str(e))

    def get_plot_configuration(self):
        return self._plot_configs

    def save_evaluation(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        self.save_kpis(output_dir)
        self.plot_paths(output_dir)
        self.plot_errors(output_dir)
        self.plot_current(output_dir)
        self.plot_wrenches(output_dir)
        self.plot_trajectories(output_dir)
        self.plot_thruster_output(output_dir)
        self.plot_error_dist(output_dir)
        self._logger.info('Evaluation stored!')

    def save_kpis(self, output_dir=None):
        if output_dir is not None:
            if not os.path.isdir(output_dir):
                self._logger.error('Invalid output directory, dir=' + str(output_dir))
                raise Exception('Invalid output directory')
        try:
            output_path = (self._output_dir if output_dir is None else output_dir)

            kpis = dict()
            kpi_labels = dict()
            for kpi in self._kpis:
                item = kpi['func']
                try:
                    value = float(item.kpi_value)
                except Exception, e:
                    value = 0.0
                kpis[item.full_tag] = value
                kpi_labels[item.full_tag] = kpi['func'].label
            with open(os.path.join(output_path, 'computed_kpis.yaml'), 'w') as kpi_file:
                yaml.dump(kpis, kpi_file, default_flow_style=False)
            self._logger.info('Calculated KPIs stored in <%s>' % os.path.join(output_path, 'computed_kpis.yaml'))

            with open(os.path.join(output_path, 'kpi_labels.yaml'), 'w') as kpi_file:
                yaml.dump(kpi_labels, kpi_file, default_flow_style=False)
            self._logger.info('KPI labels stored in <%s>' % os.path.join(output_path, 'kpi_labels.yaml'))
        except Exception, e:
            self._logger.error('Error storing KPIs file, message=' + str(e))
