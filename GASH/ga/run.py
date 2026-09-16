from json import load
import os
from tracemalloc import start
import numpy as np
import shutil
import random
import math
import multiprocessing
import json
import glob

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

def _eval_history_order(path):
    if os.path.basename(path) == "eval_history.txt":
        return (10**12, path)
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (int(stem.split("_")[-1]), path)
    except ValueError:
        return (10**12, path)


def _read_generation_history_stats(gen_dir):
    stats = {}
    order = 0
    for hist_path in sorted(glob.glob(os.path.join(gen_dir, "eval_history*.txt")), key=_eval_history_order):
        if not hist_path.endswith(".txt"):
            continue
        with open(hist_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    label = int(float(parts[0]))
                    rewards = [float(x) for x in parts[1:]]
                except ValueError:
                    continue
                if not rewards:
                    continue
                order += 1
                current = stats.get(label)
                if current is None:
                    stats[label] = {
                        "fitness": max(rewards),
                        "num_evals": len(rewards),
                        "first_order": order,
                    }
                else:
                    current["fitness"] = max(current["fitness"], max(rewards))
                    current["num_evals"] = max(current["num_evals"], len(rewards))
    return stats


def _read_final_history_labels(gen_dir):
    hist_path = os.path.join(gen_dir, "eval_history.txt")
    labels = set()
    if not os.path.exists(hist_path):
        return labels
    with open(hist_path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            try:
                labels.add(int(float(parts[0])))
            except ValueError:
                continue
    return labels


def _generation_train_iters(gen_dir, train_iters, eval_interval):
    stats = _read_generation_history_stats(gen_dir)
    full_labels = _read_final_history_labels(gen_dir)
    total = 0
    for label, row in stats.items():
        if label in full_labels:
            total += train_iters
        else:
            total += row["num_evals"] * eval_interval
    return total


def experiment_train_iters(experiment_name, train_iters, eval_interval):
    exp_dir = saved_data_path(experiment_name)
    total = 0
    for gen_dir in glob.glob(os.path.join(exp_dir, "generation_*")):
        if os.path.isdir(gen_dir):
            total += _generation_train_iters(gen_dir, train_iters, eval_interval)
    return total


def write_experiment_fitness_csv(experiment_name, train_iters, eval_interval, max_rows=None):
    exp_dir = saved_data_path(experiment_name)
    rows = {}
    for gen_dir in sorted(glob.glob(os.path.join(exp_dir, "generation_*"))):
        if not os.path.isdir(gen_dir):
            continue
        try:
            gen = int(os.path.basename(gen_dir).split("_")[-1])
        except ValueError:
            continue
        stats = _read_generation_history_stats(gen_dir)
        full_labels = _read_final_history_labels(gen_dir)
        for label, row in stats.items():
            ppo_iters = train_iters if label in full_labels else row["num_evals"] * eval_interval
            existing = rows.get(label)
            if existing is None or row["fitness"] > existing["fitness"]:
                rows[label] = {
                    "label": label,
                    "generation": gen,
                    "fitness": row["fitness"],
                    "ppo_iters": ppo_iters,
                    "status": "full" if label in full_labels else "pruned",
                }

    all_path = os.path.join(exp_dir, "final_fitness_all.csv")
    with open(all_path, "w") as f:
        f.write("label,generation,fitness,ppo_iters,status\n")
        for label in sorted(rows):
            row = rows[label]
            f.write("{},{},{},{},{}\n".format(
                row["label"], row["generation"], row["fitness"], row["ppo_iters"], row["status"]
            ))

    final_rows = list(rows.values())
    if max_rows is not None:
        final_rows = sorted(final_rows, key=lambda row: row["fitness"], reverse=True)[:max_rows]
    else:
        final_rows = sorted(final_rows, key=lambda row: row["label"])

    out_path = os.path.join(exp_dir, "final_fitness.csv")
    with open(out_path, "w") as f:
        f.write("label,generation,fitness,ppo_iters,status\n")
        for row in final_rows:
            f.write("{},{},{},{},{}\n".format(
                row["label"], row["generation"], row["fitness"], row["ppo_iters"], row["status"]
            ))
    return out_path, len(final_rows), all_path, len(rows)


def run_ga(experiment_name, structure_shape, pop_size,train_iters, num_cores,
        env_name,
        max_evaluations,
        eval_timing_arr, 
        is_pruning=False,
        scale=1,
        is_ist=False,
        resume_gen=None,
        dim_map=2,
        n_niches=128, 
        target_score=None,
        survival_rate_from_score=False,
        cm_params=cm.default_params,
        is_transfer=False,
        transfer_expr_name=None,
        transfer_gen=None,
        stop_by_train_budget=False,
        train_iter_budget=None,
        eval_interval=64,):

    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = saved_data_path(experiment_name)
    start_gen = 0
    unique_label = UniqueLabel()

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)
    if train_iter_budget is None:
        train_iter_budget = max_evaluations * train_iters

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
        temp_path = saved_data_path(experiment_name, "metadata.txt")
        
        try:
            os.makedirs(saved_data_path(experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'MAX_EVALUATIONS: {max_evaluations}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.write(f'STOP_BY_TRAIN_BUDGET: {int(stop_by_train_budget)}\n')
        f.write(f'TRAIN_ITER_BUDGET: {train_iter_budget}\n')
        f.close()

    else:
        temp_path = saved_data_path(experiment_name, "metadata.txt")
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
    archive = {}  # init archive (empty)
    curr_max=0
    params=Params(pop_size,eval_timing_arr,scale,num_cores)
    div_log=[]

    c = cm.cvt(n_niches, dim_map,
               cm_params['cvt_samples'], cm_params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')
    
    #generate a population
    if not is_continuing:
        if is_transfer==False: 
            for i in range (pop_size):
                
                temp_structure = sample_robot(structure_shape)
                while (hashable(temp_structure[0]) in population_structure_hashes):
                    temp_structure = sample_robot(structure_shape)

                structures.append(Structure(*temp_structure, unique_label.give_label(),-1))
                population_structure_hashes[hashable(temp_structure[0])] = True
                num_evaluations += 1
        else:
            for i in range(pop_size):
                save_path_structure = saved_data_path(transfer_expr_name, "generation_" + str(transfer_gen), "structure", str(i) + ".npz")
                np_data = np.load(save_path_structure)
                structure_data = []
                for key, value in np_data.items():
                    structure_data.append(value)
                structure_data = tuple(structure_data)
                population_structure_hashes[hashable(structure_data[0])] = True
                structures.append(Structure(*structure_data, i))
            num_evaluations=load_evaluation(transfer_expr_name,transfer_gen)

    #read status from file
    else:
        structures=load_archive(generation,experiment_name,filename='structures')
        archive=load_archive(generation,experiment_name)
        population_structure_hashes=load_population_hashes(generation,experiment_name)
        num_evaluations = len(list(population_structure_hashes.keys()))
        unique_label.set_label_start_for_resuming(num_evaluations)
        div_log=load_single_array_val(experiment_name,generation,'div')
        remove_only_files(experiment_name,generation)
        

    while True:

        ### UPDATE NUM SURVIORS ###	
        used_train_iters = experiment_train_iters(experiment_name, train_iters, eval_interval) if stop_by_train_budget else 0
        if stop_by_train_budget and used_train_iters >= train_iter_budget:
            out_path, num_rows, all_path, all_rows = write_experiment_fitness_csv(experiment_name, train_iters, eval_interval, max_evaluations)
            print("Reached PPO iteration budget before generation {}: {}/{}".format(generation, used_train_iters, train_iter_budget))
            print("Wrote {} selected fitness rows to {}".format(num_rows, out_path))
            print("Wrote {} total fitness rows to {}".format(all_rows, all_path))
            return

        if survival_rate_from_score:
            percent_survival= get_percent_survival_from_score(curr_max,target_score)
        else:		
            survival_eval = min(num_evaluations, max_evaluations)
            percent_survival = get_percent_survival_evals(survival_eval, max_evaluations)
        num_survivors = max(2, math.ceil(pop_size * percent_survival))


        ### MAKE GENERATION DIRECTORIES ###
        save_path_structure = saved_data_path(experiment_name, "generation_" + str(generation), "structure")
        save_path_controller = saved_data_path(experiment_name, "generation_" + str(generation), "controller")
        
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        try:
            os.makedirs(save_path_controller)
        except:
            pass

        ### SAVE POPULATION DATA ###
        for i in range (len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        ### TRAIN GENERATION

        #better parallel
        group = mp.Group()
        num_evaluated=sum([0 if structure.is_survivor else 1 for structure in structures])
        params.calc_params_interactivly(num_evaluated,scale)
        queue=multiprocessing.Queue()
        for structure in structures:

            if structure.is_survivor:
                save_path_controller_part = saved_data_path(experiment_name, "generation_" + str(generation), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                save_path_controller_part_old = saved_data_path(experiment_name, "generation_" + str(generation-1), "controller",
                    "robot_" + str(structure.label) + "_controller" + ".pt")
                
                print(f'Skipping training for {save_path_controller_part}.\n')
                
                
                try:
                    shutil.copy(save_path_controller_part_old, save_path_controller_part)
                except:
                    print(f'Error copying controller for {save_path_controller_part}.\n')
                
            else:        
                ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, structure.label),env_name,experiment_name,generation,is_pruning,params,queue)
                group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
        #initialize_start_log(experiment_name,generation,params)
        group.add_args(experiment_name,generation,params)
        group.run_jobs(num_cores,queue)

        #not parallel
        #for structure in structures:
        #    ppo.run_algo(structure=(structure.body, structure.connections), termination_condition=termination_condition, saving_convention=(save_path_controller, structure.label))

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for structure in structures:
            structure.compute_fitness()
            structure.desc=cm.calc_desc(structure.body)
            __add_to_archive(structure, structure.desc, archive, kdt)
        
        save_archive(archive,generation,experiment_name)

        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)
        
        write_output(structures,experiment_name,generation,num_evaluations)
        div_log.append(compute_diversity(structures))
        fitness_list,evaluation_list=max_fit_list_single(experiment_name,generation)
        plot_one_graph(experiment_name,generation,fitness_list,evaluation_list)
        plot_one_graph(experiment_name,generation,div_log,evaluation_list,target='div')
        save_single_array_val(div_log,experiment_name,generation,'div')

        cm.save_centroid_and_map(root_dir,experiment_name,generation,archive,n_niches)

        add_lineage(structures,experiment_name,generation)

        curr_max=structures[0].fitness
        

         ### CHECK EARLY TERMINATION ###
        used_train_iters = experiment_train_iters(experiment_name, train_iters, eval_interval) if stop_by_train_budget else 0
        if stop_by_train_budget and used_train_iters >= train_iter_budget:
            out_path, num_rows, all_path, all_rows = write_experiment_fitness_csv(experiment_name, train_iters, eval_interval, max_evaluations)
            print("Reached PPO iteration budget: {}/{}".format(used_train_iters, train_iter_budget))
            print("Wrote {} selected fitness rows to {}".format(num_rows, out_path))
            print("Wrote {} total fitness rows to {}".format(all_rows, all_path))
            return

        if not stop_by_train_budget and num_evaluations == max_evaluations:
            out_path, num_rows, all_path, all_rows = write_experiment_fitness_csv(experiment_name, train_iters, eval_interval, max_evaluations)
            print(f'Trained exactly {num_evaluations} robots')
            print("Wrote {} selected fitness rows to {}".format(num_rows, out_path))
            print("Wrote {} total fitness rows to {}".format(all_rows, all_path))
            return

        print(f'FINISHED GENERATION {generation} \n')

        ### CROSSOVER AND MUTATION ###
        # save the survivors
        survivors = structures[:num_survivors]

        #store survivior information to prevent retraining robots
        for i in range(num_survivors):
            survivors[i].is_survivor = True
    
        # for randomly selected survivors, produce children (w mutations)
        next_structures = list(survivors)
        num_children = 0
        while num_children < (pop_size - num_survivors) and (stop_by_train_budget or num_evaluations < max_evaluations):

            parent_index = random.sample(range(num_survivors), 1)
            child = mutate(survivors[parent_index[0]].body.copy(), mutation_rate = 0.1, num_attempts=50)

            if child != None and hashable(child[0]) not in population_structure_hashes:
                
                next_structures.append(Structure(*child, unique_label.give_label(),survivors[parent_index[0]].label))
                population_structure_hashes[hashable(child[0])] = True
                num_children += 1
                num_evaluations += 1

        structures = next_structures

        save_polulation_hashes(population_structure_hashes,generation,experiment_name)
        save_archive(structures,generation,experiment_name,filename='structures')

        generation += 1
