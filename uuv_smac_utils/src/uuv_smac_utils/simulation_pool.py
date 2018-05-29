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
import yaml
import logging
import sys
import datetime
from time import sleep
import random
from .utils import *
from uuv_simulation_runner import SimulationRunner
from uuv_bag_evaluation import Evaluation
from multiprocessing import Pool, Lock, Value 
from .opt_configuration import OptConfiguration

N_SIMULATION_RUNS = Value('i', 0)
N_SUCCESS = Value('i', 0)
N_CRASHES = Value('i', 0)

PROCESS_LOCK = Lock()

def add_to_crash_log(data):
    assert isinstance(data, dict)
    
    with N_SIMULATION_RUNS.get_lock():
        N_SIMULATION_RUNS.value += 1
    with N_CRASHES.get_lock():
        N_CRASHES.value += 1
    
    SIMULATION_LOGGER.error('CRASHED - Simulation failed, info=')
    for tag in data:
        SIMULATION_LOGGER.error('\t%s=%s' % (tag, data[tag]))

    SIMULATION_LOGGER.info('# simulation runs=%d' % N_SIMULATION_RUNS.value)
    SIMULATION_LOGGER.info('\tSUCCESS=%d' % N_SUCCESS.value)
    SIMULATION_LOGGER.info('\tCRASHED=%d' % N_CRASHES.value)

def add_to_run_log(data):
    assert isinstance(data, dict)

    with N_SIMULATION_RUNS.get_lock():
        N_SIMULATION_RUNS.value += 1
    with N_SUCCESS.get_lock():
        N_SUCCESS.value += 1

    SIMULATION_LOGGER.info('SUCCESS - Simulation finished successfully, info=')
    for tag in data:
        SIMULATION_LOGGER.info('\t%s=%s' % (tag, data[tag]))    

    SIMULATION_LOGGER.info('# simulation runs=%d' % N_SIMULATION_RUNS.value)
    SIMULATION_LOGGER.info('\tSUCCESS=%d' % N_SUCCESS.value)
    SIMULATION_LOGGER.info('\tCRASHED=%d' % N_CRASHES.value)

def run_simulation(task):
    random.seed()
    sleep(random.random())
    opt_config = OptConfiguration.get_instance()

    SIMULATION_LOGGER.info('Starting simulation for task <%s>...' % task)
    SIMULATION_LOGGER.info('\tParameters=' + str(opt_config.params))
    SIMULATION_LOGGER.info('\tPartial results root directory=' + opt_config.results_dir)
    SIMULATION_LOGGER.info('\tRecord all partial results? ' + str(opt_config.record_all))

    runner = None
    sim_eval = None
    try:        
        runner = SimulationRunner(
            opt_config.params, task, opt_config.results_dir, opt_config.record_all)
        runner.run(opt_config.params)
        sleep(random.random())

        recording_dirname = os.path.dirname(runner.recording_filename)

        has_recording = False
        for item in os.listdir(recording_dirname):
            if item.endswith('.bag') or item.endswith('.bag.active'):
                has_recording = True
                break

        if has_recording is False:
            SIMULATION_LOGGER.error('No recording generated for task <%s>' % task)
        else:           
            has_recording = False     
            for _ in range(30):                    
                for item in os.listdir(recording_dirname):                        
                    if item.endswith('.bag'):
                        has_recording = True
                        break
                    else:
                        sleep(0.1)
                   
        if not has_recording:
            raise Exception('No recording generated for task <%s>, file=%s' % (task, runner.recording_filename))
    except Exception, e:
        SIMULATION_LOGGER.error('Error occurred in this iteration, setting simulation status to CRASHED for task <%s>, message=%s' % (task, str(e)))
        status = SIM_CRASHED
        partial_cost = 1e7        

        add_to_crash_log(dict(
            status=status,
            timestamp=str(datetime.datetime.now().isoformat()),
            message=str(e),
            task=str(task)))
        
        output = dict(task=str(task), status=status, cost=partial_cost, sim_time=None, message=str(e))

        if runner is not None:
            del runner            
        return output
    else:
        SIMULATION_LOGGER.info('Simulation finished, task=%s' % task)

    try:
        PROCESS_LOCK.acquire()
        
        time_offset = 0.0
        if opt_config.evaluation_time_offset is not None:
            time_offset = max(0.0, opt_config.evaluation_time_offset)
        
        SIMULATION_LOGGER.info('Start evaluation of the results')
        SIMULATION_LOGGER.info('\tTime offset for KPI evaluation[s]=' + str(time_offset))
        SIMULATION_LOGGER.info('\tResults files directory=' + runner.current_sim_results_dir)
        SIMULATION_LOGGER.info('\tROS bag file=' + runner.recording_filename)
        sim_eval = Evaluation(runner.recording_filename,
                              runner.current_sim_results_dir,
                              time_offset=time_offset)

        SIMULATION_LOGGER.info('Evaluation finished')

        sim_eval.compute_kpis()
        
        if opt_config.store_kpis_only:
            sim_eval.save_kpis()
            SIMULATION_LOGGER.info('Store KPIs only')
        else:
            sim_eval.save_evaluation()
            SIMULATION_LOGGER.info('Store KPIs and graphs')

        SIMULATION_LOGGER.info('Calculating cost function')
        partial_cost = opt_config.compute_cost_fcn(sim_eval.get_kpis())
        
        if partial_cost < 0:
            raise Exception('Cost function returned value lower than zero')
        
        sim_time = float(runner.timeout - time_offset)
        
        opt_config.cost_fcn.save(runner.current_sim_results_dir)

        status = SIM_SUCCESS
        output = dict(
            timestamp=str(datetime.datetime.now().isoformat()),
            status=status,
            cost=float(partial_cost),
            sim_time=sim_time, 
            task=task)

        add_to_run_log(output)

        SIMULATION_LOGGER.info('Cost function=' + str(partial_cost))            
        SIMULATION_LOGGER.info('Simulation timeout=%.2f s' % sim_time)       

        with open(os.path.join(runner.current_sim_results_dir, 'smac_result.yaml'), 'w+') as smac_file:
            yaml.dump(output, smac_file, default_flow_style=False)

        sleep(random.random())
    except Exception, e:
        SIMULATION_LOGGER.error(
            'Error occurred in this simulation evaluation, '
            'setting simulation status to CRASHED for task '
            '<%s>, message=%s' % (task, str(e)))
        status = SIM_CRASHED
        partial_cost = 1e7        

        add_to_crash_log(dict(
            status=status,
            timestamp=str(datetime.datetime.now().isoformat()),
            message=str(e),
            task=str(task)))

        output = dict(status=status, cost=partial_cost, sim_time=None, message=str(e), task=str(task))
        PROCESS_LOCK.release()

        if runner is not None:
            del runner
        if sim_eval is not None:
            del sim_eval

        return output        
    else:
        if runner is not None:
            del runner
        if sim_eval is not None:
            del sim_eval
            
        PROCESS_LOCK.release()        
        sleep(random.random())
        return output        

def start_simulation_pool(max_num_processes=None, tasks=None, log_filename=None):
    init_logger(log_filename)

    opt_config = OptConfiguration.get_instance()
    opt_config.print_params()

    num_processes = max_num_processes
    if num_processes is None:
        num_processes = opt_config.max_num_processes

    SIMULATION_LOGGER.info('Starting simulation pool, num_processes=%d' % num_processes)

    try:
        thread_pool = Pool(processes=num_processes)
    except KeyboardInterrupt:
        SIMULATION_LOGGER.info('Key interrupt! Killing all processes...')
        thread_pool.terminate()
        return None, None    

    task_list = tasks
    if tasks is None:
        task_list = opt_config.tasks
    output = thread_pool.map(run_simulation, task_list)

    thread_pool.close()
    thread_pool.join()

    del thread_pool

    failed_tasks = list()

    for item in output:
        if item['status'] == SIM_CRASHED:
            failed_tasks.append(item['task'])

    SIMULATION_LOGGER.info('Ending simulation pool, # failed tasks=%d' % len(failed_tasks))

    return output, failed_tasks
    