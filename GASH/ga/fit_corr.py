import numpy as np
import os
from pathlib import Path
import shutil
from copy import deepcopy
from socket import gethostname

import random
from utils.algo_utils import *
from evogym import is_connected, has_actuator, get_full_connectivity,hashable
from math import floor
import multiprocessing
from ppo import run_ppo
from utils.algo_utils import *
import utils.mp_group as mp

root_dir = os.path.dirname(Path().resolve())
root_dir=os.path.join(root_dir, 'examples')

def find_gen(save_dir,max_score_index):
    for i in range(1000):
        log_dir = os.path.join(save_dir,'generation_'+str(i),"structure", str(max_score_index) + ".npz")
        if os.path.exists(log_dir):
            return i
    print('FileNotFoundError : cannot find directory')
        

def find_best_strcture_per_seed(expr_name,last_gen,seed):
    log_dir=expr_name+'_seed:'+str(seed)
    save_dir = os.path.join(root_dir, 'saved_data',log_dir)
    log_dir = os.path.join(save_dir,'generation_'+str(last_gen),'output.txt')
    with open(log_dir) as f:
        for line in f:
            max_score_index=int(line.split()[0])
            max_score=float(line.split()[1])
            break
    g=find_gen(save_dir,max_score_index)
    print(g)
    save_path_structure = os.path.join(save_dir, "generation_" + str(g), "structure", str(max_score_index) + ".npz")
    np_data = np.load(save_path_structure)
    structure_data = []
    for key, value in np_data.items():
        structure_data.append(value)
    structure_data = tuple(structure_data)
    return Structure(*structure_data, 0,-1),max_score

def find_best_structure(load_expr_name,last_gen):
    best=None
    curr_max=-100
    for seed in range(101,108):
        structure,max_score=find_best_strcture_per_seed(load_expr_name,last_gen,seed)
        if max_score>curr_max:
            best=structure
            curr_max=max_score
            print('curr max : seed={}, score={}'.format(seed,max_score))
    return best

def load_best_structure(expr_name):
    save_path = os.path.join(root_dir, "best_structures", expr_name)
    with open(save_path+"/best",'rb') as f:
        return pickle.load(f)



def mutate_distance_k(child,parent,k,population_structure_hashes,unique_label):
    while calc_edit_distance(child,parent)<k:
        val=random.randrange(5)
        i=random.randrange(5)
        j=random.randrange(5)
        child[i][j]=val
    
    if is_connected(child) and has_actuator(child) and hashable(child) not in population_structure_hashes:
        population_structure_hashes[hashable(child)] = True
        return Structure(child, get_full_connectivity(child), unique_label.give_label(),-1)
    else:
        if is_connected(child) is False:
            print('cannot connected')
        if has_actuator(child) is False:
            print('do not have actuator')
        if hashable(child) in population_structure_hashes:
            print("Not Hashable. Passed when k={}".format(k))
        return None

def calc_fit_corr(
    experiment_name,
    load_expr_name,
    last_gen,
    train_iters, 
    num_cores,
    env_name,
    num_per_k,
    max_k,
    p,
    batch_size):
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
    unique_label = UniqueLabel()

    ### DEFINE TERMINATION CONDITION ###    
    tc = TerminationCondition(train_iters)

    ist_hostname_list=['login000','login001','big000','big001']

    try:
        os.makedirs(home_path)
    except:
        print(f'THIS EXPERIMENT ({experiment_name}) ALREADY EXISTS')
        if gethostname() in ist_hostname_list:
            print('executed in ist. rmtree and execute experiment\n')
            shutil.rmtree(home_path)
        else:
            print("Override? (y/n): ", end="")
            ans = input()
            if ans.lower() == "y":
                shutil.rmtree(home_path)
                print()
            else:
                return

    structures=[]
    population_structure_hashes = {}

    parent=load_best_structure(load_expr_name)

    for k in range(1,max_k+1):
        valid=0
        while valid<num_per_k:
            child=mutate_distance_k(parent.body.copy(),parent.body,k,population_structure_hashes,unique_label)
            if child is not None:
                child.distance=k
                structures.append(child)
                for i in range(1,p):
                    structures.append(deepcopy(child))
                    structures[len(structures)-1].torch_seed=i+1
                valid+=1
    
    ### MAKE GENERATION DIRECTORIES ###
    save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "structure")
    save_path_controller = os.path.join(root_dir, "saved_data", experiment_name, "controller")
    
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

    num_batch=len(structures)//batch_size+1
    print('num of batch is {}'.format(num_batch))
                


    group = mp.Group()
    queue=multiprocessing.Queue()

    for batch_index in range(num_batch):
        print('start batch number : {}'.format(batch_index))
        start_index=batch_size*batch_index
        fin_index=batch_size*(batch_index+1) if batch_index+1<num_batch else len(structures)

        group = mp.Group()
        queue=multiprocessing.Queue()
        for structure in structures[start_index:fin_index]:   
            ppo_args = ((structure.body, structure.connections), tc, (save_path_controller, (structure.label*p+structure.torch_seed-1)),env_name,None,None,False,None,queue,structure.torch_seed)
            group.add_job(run_ppo, ppo_args, callback=structure.set_reward)
        group.run_jobs(num_cores,queue)

    for structure in structures:
        structure.compute_fitness()

    ## save output.txt ##
    temp_path = os.path.join(root_dir, "saved_data", experiment_name, "output.txt")
    f = open(temp_path, "w")

    out = ""
    for structure in structures:
        out += str(structure.label) + "\t\t" + str(structure.distance) + "\t\t"+str(structure.fitness) + "\n"
    f.write(out)
    f.close()

    print(f'FINISHED \n')




