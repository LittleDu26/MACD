from json import load
import os
import numpy as np
import shutil
import random
import math
import multiprocessing

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from pyME.map_elites.single_cvt import __add_to_archive
from pyME.map_elites import common as cm
from sklearn.neighbors import KDTree
from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import *
from utils.pruning_params import Params



def run_ga_tournament(experiment_name, structure_shape, pop_size,train_iters,num_cores,
    env_name,
    max_evaluations, 
    eval_timing_arr, 
    is_pruning=False,
    scale=1,
    is_ist=False,
    resume_gen=None,
    dim_map=2,
    n_niches=128,
    params=cm.default_params,
    ):

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    start_gen = 0
    unique_label = UniqueLabel()

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)

    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        if is_ist:
            print('this experiment is launched in ist, continue from gen:'+str(resume_gen))
            start_gen=resume_gen
            is_continuing=True
        else:
            print("Override? (y/n/c): ", end="")
            ans = input()
            if ans.lower() == "y":
                shutil.rmtree(home_path)
                print()
            elif ans.lower() == "c":
                print("Enter gen to start training on (0-indexed): ", end="")
                start_gen = int(input())
                is_continuing = True
                print()
            else:
                return

    ### STORE META-DATA ##
    if not is_continuing:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        
        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")
        f = open(temp_path, "r")
        count = 0
        for line in f:
            if count == 0:
                pop_size = int(line.split()[1])
            if count == 1:
                structure_shape = (int(line.split()[1]), int(line.split()[2]))
            if count == 2:
                max_evaluations = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' + 
            f'max evals: {max_evaluations}, train iters {train_iters}.')
        
        f.close()

    ### GENERATE // GET INITIAL POPULATION ###
    structures = []
    population_structure_hashes = {}
    num_evaluations = 0
    generation = start_gen
    params=Params(pop_size,eval_timing_arr,scale,num_cores)
    div_log=[]

    # random initialization
    if not is_continuing:
        # initialize robot
        for i in range(pop_size):
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, unique_label.give_label(),-1))
            population_structure_hashes[hashable(temp_structure[0])] = True

    else:  
        structures=load_archive(generation,experiment_name,filename='structures')
        population_structure_hashes=load_population_hashes(generation,experiment_name)
        num_evaluations=calc_curr_evaluation(generation,pop_size,pop_size)
        unique_label.set_label_start_for_resuming(num_evaluations+pop_size)
        div_log=load_single_array_val(experiment_name,generation,'div')
        assert len(structures) == 2*pop_size
        
    
   
    while (num_evaluations < max_evaluations):
         ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range(pop_size):
            temp_path = os.path.join(save_path_structure, str(structures[len(structures)-pop_size+i].label))
            np.savez(temp_path, structures[len(structures)-pop_size+i].body, structures[len(structures)-pop_size+i].connections)

    
        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        params.calc_params_interactivly(pop_size,scale)
        queue=multiprocessing.Queue()
        for structure in structures[len(structures)-pop_size:]:
            ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),env_name,experiment_name,generation,is_pruning,params,queue)
            group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
            num_evaluations+=1
        group.run_jobs(num_cores,queue)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures[len(structures)-pop_size:]:
            structure.compute_fitness()

        save_evaluated_score(experiment_name,generation,structures[len(structures)-pop_size:])

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        structures = structures[:pop_size]

        #SAVE RANKING TO FILE
        write_output(structures,experiment_name,generation,num_evaluations)
        div_log.append(compute_diversity(structures))
        fitness_list,evaluation_list=max_fit_list_single(experiment_name,generation)
        plot_one_graph(experiment_name,generation,fitness_list,evaluation_list)
        plot_one_graph(experiment_name,generation,div_log,evaluation_list,target='div')
        save_single_array_val(div_log,experiment_name,generation,'div')

        #SAVE LINEAGE TO FILE
        add_lineage(structures,experiment_name,generation)

         ### CHECK EARLY TERMINATION ###
        if num_evaluations >= max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return

        print(f'FINISHED GENERATION {generation} \n')
        
        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < pop_size and num_evaluations+num_children<max_evaluations:

            parent_index = tournament_selection(structures,pop_size)
            child = mutate(structures[parent_index].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                # overwrite structures array w new child
                structures.append(Structure(*child, unique_label.give_label(),structures[parent_index].label))
                population_structure_hashes[hashable(child[0])] = True
                num_children+=1
        
        
        save_polulation_hashes(population_structure_hashes,generation,experiment_name)
        save_archive(structures,generation,experiment_name,filename='structures')
        # write structure pkl file(structure length = 2* popsize)
       
        generation += 1