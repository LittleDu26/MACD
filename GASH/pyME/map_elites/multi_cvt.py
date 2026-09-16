from random import random
from typing import Tuple
import numpy as np
import shutil
import sys, os,itertools

curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '../..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
from sklearn.neighbors import KDTree

from pyME.map_elites import common as cm
from utils.algo_utils import mutate, TerminationCondition, Structure, UniqueLabel
from evogym import sample_robot, hashable
from ppo import run_ppo
import utils.mp_group as mp

population_structure_hashes = {}
unique_label = UniqueLabel()


def __add_to_archive(x, centroid, archive, kdt, pareto_max):
    niche_index = kdt.query([centroid], k=1)[1][0][0]
    niche = kdt.data[niche_index]
    n = cm.make_hashable(niche)
    x.centroid = n
    if n in archive:
        Judge_dominated(x, n, archive, pareto_max)

    else:
        archive[n] = []
        archive[n].append(x)

class MyProblem():

    def __init__(self,tc,num_cores,env_name1,env_name2) -> None:
        self.tc = tc
        self.num_cores=num_cores
        self.env_name1=env_name1
        self.env_name2=env_name2

    def _evaluate(self,X,save_path_controller1,save_path_controller2):
            
        self.RunOneEnv(X,self.env_name1,save_path_controller1,1)
        self.RunOneEnv(X,self.env_name2,save_path_controller2,2)

    def RunOneEnv(self,X,env_name,save_path_controller,env_index):
        group = mp.Group()
        for x in X:
            ppo_args = ((x.body, x.connections), self.tc, (save_path_controller, x.label),env_name)
            if env_index==1:
                group.add_job(run_ppo, ppo_args, callback=x.set_reward)
            elif env_index==2:
                group.add_job(run_ppo, ppo_args, callback=x.set_reward2)
        #exit(1)
        group.run_jobs(self.num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        for i,x in enumerate(X):
            if env_index==1:
                x.compute_fitness()
            elif env_index==2:
                x.compute_fitness2()
            x.desc=cm.calc_desc(x.body)


def Judge_dominated(x: Structure, n, archive, pareto_max):
    # removeが成功してるかのチェック入れる
    dominated = []
    is_dominant = False
    for item in archive[n]:
        if x.fitness > item.fitness and x.fitness2 > item.fitness2:
            dominated.append(item)
        elif x.fitness < item.fitness and x.fitness2 < item.fitness2:
            is_dominant = True

    assert len(dominated) == 0 or is_dominant == False, print(
        'RunTimeError: x dominates some object and dominated. It is imcompatible.')

    pre_archive_n_length = len(archive[n])
    if is_dominant:
        return
    elif len(dominated) > 0:
        for item in dominated:
            archive[n].remove(item)
    assert pre_archive_n_length - len(dominated) == len(archive[n]), print('RunTimeError:remove process failed')
    assert len(archive[n]) <= pareto_max, print('RunTimeError: length of cell exceeds pareto_max')
    
    if len(archive[n]) == pareto_max:
        archive[n].pop(np.random.randint(len(archive[n])))

    archive[n].append(x)
    return


def make_save_path(expr_name, generation):
    save_path_controller1 = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(generation),
                                         "controller1")
    try:
        os.makedirs(save_path_controller1)
    except:
        pass

    save_path_controller2 = os.path.join(root_dir, "saved_data", expr_name, "generation_" + str(generation),
                                         "controller2")
    try:
        os.makedirs(save_path_controller2)
    except:
        pass
    return save_path_controller1, save_path_controller2


# map-elites algorithm (CVT variant)
def run_MOME(experiment_name, structure_shape, pop_size, total_generation,
            train_iters, num_cores, env_name1, env_name2, n_samples, batch_size, p_mut, pareto_max,
            dim_map,
            n_niches,
            params=cm.default_params):
    """CVT MAP-Elites
       Vassiliades V, Chatzilygeroudis K, Mouret JB. Using centroidal voronoi tessellations to scale up the multidimensional archive of phenotypic elites algorithm. IEEE Transactions on Evolutionary Computation. 2017 Aug 3;22(4):623-30.

       Format of the logfile: evals archive_size max mean median 5%_percentile, 95%_percentile

    """
    ### STARTUP: MANAGE DIRECTORIES ###
    home_path = os.path.join(root_dir, "saved_data", experiment_name)
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
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "metadata.txt")

        try:
            os.makedirs(os.path.join(root_dir, "saved_data", experiment_name))
        except:
            pass

        f = open(temp_path, "w")
        f.write(f'POP_SIZE: {pop_size}\n')
        f.write(f'STRUCTURE_SHAPE: {structure_shape[0]} {structure_shape[1]}\n')
        f.write(f'TOTAL_GENERATION: {total_generation}\n')
        f.write(f'TRAIN_ITERS: {train_iters}\n')
        f.write(f'ENV_NAMES1: {env_name1}\n')
        f.write(f'ENV_NAMES2: {env_name2}\n')
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
                total_generation = int(line.split()[1])
            if count == 3:
                train_iters = int(line.split()[1])
                tc.change_target(train_iters)
            count += 1

        print(f'Starting training with pop_size {pop_size}, shape ({structure_shape[0]}, {structure_shape[1]}), ' +
              f'max evals: {total_generation}, train iters {train_iters}.')

        f.close()
    # setup the parallel processing pool
    myProblem = MyProblem(tc, num_cores, env_name1, env_name2)

    # create the CVT
    c = cm.cvt(n_niches, dim_map,
               params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')

    archive = {}  # init archive (empty)
    generation = 0  # number of evaluations since the beginning
    total_evaluation=0
    b_generation = 0  # number evaluation since the last dump

    # main loop
    while (generation < total_generation):
        save_path_controller1, save_path_controller2 = make_save_path(experiment_name, generation)
        # random initialization
        if generation == 0:
            # initialize robot
            X = np.full(n_samples, None, dtype=object)
            total_evaluation+=n_samples

            for i in range(n_samples):
                temp_structure = sample_robot(structure_shape)
                while (hashable(temp_structure[0]) in population_structure_hashes):
                    temp_structure = sample_robot(structure_shape)

                X[i] = Structure(*temp_structure, unique_label.give_label())
                population_structure_hashes[hashable(temp_structure[0])] = True

        else:  # variation/selection loop
            keys = list(archive.keys())
            n_mut = int(batch_size * p_mut)
            n_cross = int(batch_size * (1 - p_mut))
            X = np.full(n_mut + n_cross, None, dtype=object)
            total_evaluation+=batch_size
            
            # mutation
            rand = np.random.randint(len(keys), size=n_mut)
            for i in range(0, n_mut):
                # parent selection
                k = np.random.randint(len(archive[keys[rand[i]]]))
                x = archive[keys[rand[i]]][k]
                # copy & add variation
                child = None

                while child == None or hashable(child[0]) in population_structure_hashes:
                    child = mutate(x.body.copy(), mutation_rate=0.1, num_attempts=50)

                # overwrite structures array w new child
                X[i] = Structure(*child, unique_label.give_label())
                population_structure_hashes[hashable(child[0])] = True

            # crossover
            rand1 = np.random.randint(len(keys), size=batch_size)
            rand2 = np.random.randint(len(keys), size=batch_size)
            for i in range(0, n_cross):
                # parent selection
                k1 = np.random.randint(len(rand1[i]))
                k2 = np.random.randint(len(rand2[i]))
                x = archive[keys[rand1[i]]][k1]
                y = archive[keys[rand2[i]]][k2]
                # copy & add variation
                child = None

                # cross overは後で書く
                while child == None or hashable(child[0]) in population_structure_hashes:
                    child = mutate(x.body.copy(), mutation_rate=0.1, num_attempts=50)

                # overwrite structures array w new child
                X[i] = Structure(*child, unique_label.give_label())
                population_structure_hashes[hashable(child[0])] = True

            print("n_mut",n_mut,"\n n_cross",n_cross,"\n X",X)
        # evaluation of the fitness for to_evaluate
        myProblem._evaluate(X, save_path_controller1, save_path_controller2)
        # natural selection
        for x in X:
            __add_to_archive(x, x.desc, archive, kdt,pareto_max)

        ######################save data per generation##########################

        save_path_structure = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation),
                                           "structure")
        try:
            os.makedirs(save_path_structure)
        except:
            pass

        structures = [structure for structure in (archive[key] for key in archive)]
        structures = itertools.chain.from_iterable(structures)
        structures = sorted(structures, key=lambda structure: structure.fitness, reverse=True)

        ### SAVE POPULATION DATA ###
        for i in range(len(structures)):
            temp_path = os.path.join(save_path_structure, str(structures[i].label))
            np.savez(temp_path, structures[i].body, structures[i].connections)

        # SAVE RANKING TO FILE
        temp_path = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(generation), "output.txt")
        f = open(temp_path, "w")

        out = ""
        for structure in structures:
            out += str(structure.label) + "\t\t" + str(structure.fitness) + "\t\t" + str(
                structure.fitness2) + "\n"
        out+="total evaluation: "+str(total_evaluation)+"\n"
        f.write(out)
        f.close()

        print(f'FINISHED GENERATION {generation}\n')
        
        '''
        if generation % 5 == 0:
            plot = Scatter(title="Objective Space", labels="f")
            front = algorithm.pop.get("F")
            plot.add(-front)
            temp_path = os.path.join(root_dir, "saved_data", self.experiment_name,
                                     "pareto_front_" + str(generation) + ".pdf")
            plot.save(temp_path)
        '''

        ### update generation ###
        generation += 1
        b_generation += 1
