import os
import numpy as np
import shutil
import random
import math
import csv
import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
from .CSVReporter import CustomReporter
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import get_percent_survival_evals, ga_mutate, TerminationCondition, Structure
import torch
def run_ga(experiment_name, structure_shape, pop_size, max_ganeration,max_evaluations, train_iters, num_cores,device_num,save_path,seed=101):
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = save_path
    start_gen = 0

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)


    is_continuing = False    
    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
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
        temp_path = os.path.join(home_path, "metadata.txt")
        
        try:
            os.makedirs(home_path)
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'ENV_NAME: {experiment_name}\n')
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_GENERATIONs: {max_ganeration}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.close()

    else:
        temp_path = os.path.join(home_path, "metadata.txt")
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
    generation = 0
    logger = CustomReporter(home_path)
    #generate a population
    if not is_continuing: 
        for i in range (pop_size):
            
            temp_structure = sample_robot(structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(structure_shape)

            structures.append(Structure(*temp_structure, i))
            population_structure_hashes[hashable(temp_structure[0])] = True
            num_evaluations += 1

    #read status from file
    else:
        for g in range(start_gen+1):
            for i in range(pop_size):
                save_path_structure = os.path.join(home_path, "generation" + str(g), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                # only a current structure if last gen
                if g == start_gen:
                    structures.append(Structure(*structure_data, i))
        num_evaluations = len(list(population_structure_hashes.keys()))
        generation = start_gen


    while True:
        logger.start_generation(generation)
        ### UPDATE NUM SURVIORS ###
        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))

        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = os.path.join(home_path, "generation" + str(generation), "structure")
        save_path_controller = os.path.join(home_path, "generation" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### TRAIN GENERATION
        tmp_inds = []
        fits = []
        #better parallel
        group = mp.Group()
        for structure in structures:
            if structure.is_survivor:
                save_path_controller_part = os.path.join(home_path, "generation" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = os.path.join(home_path, "generation" + str(generation-1), "controller",
                    "robot_" + str(structure.prev_gen_label) + "_controller" + ".pt")
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error coppying controller from {save_path_controller_part_old}.\n')
            else:
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),experiment_name,device_num,seed)
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
                tmp_inds.append(structure)

        group.run_jobs(num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()
        for tmp_structure in tmp_inds:
            fits.append(tmp_structure.compute_fitness())
        optima_path = os.path.join(home_path, 'pop.csv')
        with open(optima_path, 'a+', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            for fi in range(len(fits)):
                writer.writerow([generation,fi, fits[fi]])

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        logger.end_generation(structures)

        ### SAVE POPULATION DATA ###
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### CHECK EARLY TERMINATION ###
        if num_evaluations == max_evaluations:
            print(f'Trained exactly {num_evaluations} robots')
            return
        print(f'FINISHED GENERATION {generation} - SEE TOP {round(percent_survival*100)}% of Survivors:\n')

        ### MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            structures[i].is_survivor = True
            structures[i].prev_gen_label = structures[i].label
            structures[i].label = i

        # for randomly selected survivors, produce children (w mutations)
        num_children = 0
        while num_children < (pop_size - num_survivors) and num_evaluations < max_evaluations:

            parent_index = random.sample(range(num_survivors), 1)
            child = ga_mutate(survivors[parent_index[0]].body.copy(),  mutation_rate = 0.1, num_attempts=50)

            if child is not None and hashable(child[0]) not in population_structure_hashes:
                # overwrite structures array w new child
                structures[num_survivors + num_children] = Structure(*child, num_survivors + num_children)
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = structures[:num_children+num_survivors]

        generation += 1