import os
import numpy as np
import shutil
import random
import math

import sys
curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
population_structure_hashes = {}
generation=0
save_path_controller1=None
save_path_controller2=None
curr_evaluation=0

from ppo import run_ppo
from evogym import sample_robot, hashable
import utils.mp_group as mp
from utils.algo_utils import *
from make_gifs_multi import Job
from pyME.map_elites.single_cvt import __add_to_archive
from pyME.map_elites import common as cm
from sklearn.neighbors import KDTree
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from pymoo.factory import get_visualization
from pymoo.core.problem import Problem
from pymoo.visualization.scatter import Scatter

unique_label=UniqueLabel()


class MyProblem(Problem):

    def __init__(
        self,
        algorithm,
        tc,
        num_cores,
        env_name1,
        env_name2,
        novelty_archive,
        archive,
        kdt,
        __add_to_archive,
        experiment_name,
        is_two_env_parallel=False,
        is_NSGC=False) -> None:

        super().__init__(n_var=1, n_obj=2, n_constr=0)
        self.algorithm=algorithm
        self.tc = tc
        self.num_cores=num_cores
        self.env_name1=env_name1
        self.env_name2=env_name2
        self.novelty_archive=novelty_archive
        self.archive=archive
        self.kdt=kdt
        self.__add_to_archive = __add_to_archive
        self.experiment_name=experiment_name
        self.is_two_env_parallel=is_two_env_parallel
        self.is_NSGC=is_NSGC

    def _evaluate(self, X, out, *args, **kwargs):
        if self.is_two_env_parallel:
            fitness1,fitness2=self.RunParallelEnv(X,self.env_name1,self.env_name2)
        elif self.is_NSGC:
            fitness1,fitness2=self.NSGC(X,self.env_name1,save_path_controller1)
        else:
            fitness1=self.RunOneEnv(X,self.env_name1,save_path_controller1,1)
            fitness2=self.RunOneEnv(X,self.env_name2,save_path_controller2,2)
        out["F"] = np.column_stack([fitness1, fitness2])
    
    def RunOneEnv(self,X,env_name,save_path_controller,env_index):
        global curr_evaluation

        group = mp.Group()
        for x in X:
            ppo_args = ((x[0].body, x[0].connections), self.tc, (save_path_controller, x[0].label),env_name)
            if env_index==1:
                group.add_job(run_ppo, ppo_args, callback=x[0].set_reward)
            elif env_index==2:
                group.add_job(run_ppo, ppo_args, callback=x[0].set_reward2)
            curr_evaluation+=1
        group.run_jobs(self.num_cores)

        ### COMPUTE FITNESS, SORT, AND SAVE ###
        f=np.full((X.shape[0],1),None,dtype=object)
        for i,x in enumerate(X):
            if env_index==1:
                x[0].compute_fitness()
                f[i][0]=-x[0].fitness
            elif env_index==2:
                x[0].compute_fitness2()
                f[i][0]=-x[0].fitness2
        return f
    
    def NSGC(self,X,env_name,save_path_controller):
        global curr_evaluation
        global generation

        group = mp.Group()
        for x in X:
            ppo_args = ((x[0].body, x[0].connections), self.tc, (save_path_controller, x[0].label),env_name)
            group.add_job(run_ppo, ppo_args, callback=x[0].set_reward)
            curr_evaluation+=1
        group.run_jobs(self.num_cores)
        save_evaluated_structures(self.experiment_name,generation,X.ravel())

        f1=np.full((X.shape[0],1),None,dtype=object)
        f2=compute_novelty(X,self.novelty_archive)
        for i,x in enumerate(X):
            f1[i][0]=-x[0].fitness
             #for saving in output.txt
            x[0].fitness2=f2[i][0]
            f2[i][0]=-f2[i][0]
            x[0].desc=cm.calc_desc(x[0].body)
            self.__add_to_archive(x[0],x[0].desc,self.archive,self.kdt)
        save_evaluated_score(self.experiment_name,generation,X.ravel())
        return f1,f2

    def RunParallelEnv(self,X,env_name1,env_name2):
        global curr_evaluation
        group = mp.Group()
        for x in X:
            ppo_args1 = ((x[0].body, x[0].connections), self.tc, (save_path_controller1, x[0].label),env_name1)
            ppo_args2 = ((x[0].body, x[0].connections), self.tc, (save_path_controller2, x[0].label),env_name2)
            group.add_job(run_ppo, ppo_args1, callback=x[0].set_reward)
            group.add_job(run_ppo, ppo_args2, callback=x[0].set_reward2)
            curr_evaluation+=2
        group.run_jobs(self.num_cores)

        f1=np.full((X.shape[0],1),None,dtype=object)
        f2=np.full((X.shape[0],1),None,dtype=object)
        for i,x in enumerate(X):
            x[0].compute_fitness()
            f1[i][0]=-x[0].fitness
            x[0].compute_fitness2()
            f2[i][0]=-x[0].fitness2
        return f1,f2

def create_controller_dir(experiment_name,gen,is_NSGC):
    global save_path_controller1
    global save_path_controller2

    if is_NSGC:
        save_path_controller1 = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(gen), "controller")
        try:
            os.makedirs(save_path_controller1)
        except:
            pass
    else:
        save_path_controller1 = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(gen), "controller1")
        try:
            os.makedirs(save_path_controller1)
        except:
            pass
        save_path_controller2 = os.path.join(root_dir, "saved_data", experiment_name, "generation_" + str(gen), "controller2")
        try:
            os.makedirs(save_path_controller2)
        except:
            pass


class MySampling(Sampling):
    def __init__(self,structure_shape,experiment_name,is_NSGC,is_continuing=False):
        super().__init__()
        self.structure_shape=structure_shape
        self.experiment_name=experiment_name
        self.is_continuing=is_continuing
        self.is_NSGC=is_NSGC

    def _do(self, problem, n_samples, **kwargs):
        create_controller_dir(self.experiment_name,generation,self.is_NSGC)

        X = np.full((n_samples, 1), None, dtype=object)

        for i in range(n_samples):
            temp_structure = sample_robot(self.structure_shape)
            while (hashable(temp_structure[0]) in population_structure_hashes):
                temp_structure = sample_robot(self.structure_shape)

            X[i, 0] = Structure(*temp_structure, unique_label.give_label(),-1)
            population_structure_hashes[hashable(temp_structure[0])] = True

        return X

class MyCrossover(Crossover):
    def __init__(self):

        # define the crossover: number of parents and number of offsprings
        super().__init__(2, 2)

    def _do(self, problem, X, **kwargs):

        return X

class MyMutation(Mutation):
    def __init__(self):
        super().__init__()

    def _do(self, problem, X, **kwargs):

        # for each individual
        for i in range(len(X)):
            child=None

            while child ==None or hashable(child[0]) in population_structure_hashes:
                child = mutate(X[i,0].body.copy(), mutation_rate = 0.1, num_attempts=50)

            # overwrite structures array w new child
            X[i,0] = Structure(*child, unique_label.give_label(),X[i,0].label)
            population_structure_hashes[hashable(child[0])] = True

        return X

class MyCallback(Callback):
        def __init__(self,experiment_name,is_NSGC,archive,n_niches) -> None:
            super().__init__()
            self.experiment_name = experiment_name
            self.is_NSGC=is_NSGC
            self.archive=archive
            self.n_niches=n_niches

        def notify(self, algorithm):
            global generation
            global save_path_controller1
            global save_path_controller2
            global curr_evaluation

            save_polulation_hashes(population_structure_hashes,generation,self.experiment_name)
            
            structures=algorithm.pop.get("X")
            structures = sorted(structures, key=lambda structure: structure[0].fitness, reverse=True)
            
            #SAVE RANKING TO FILE
            temp_path = os.path.join(root_dir, "saved_data", self.experiment_name, "generation_" + str(generation), "output.txt")
            f = open(temp_path, "w")

            out = ""
            for structure in structures:
                out += str(structure[0].label) + "\t\t" + str(structure[0].fitness) + "\t\t" + str(structure[0].fitness2) + "\n"
            out+="current evaluation: "+str(curr_evaluation)+"\n"
            f.write(out)
            f.close()
            plot_graph(self.experiment_name,generation,True)
            cm.save_centroid_and_map(root_dir,self.experiment_name,generation,self.archive,self.n_niches)
            
            #SAVE LINEAGE TO FILE
            add_lineage(np.ravel(structures),self.experiment_name,generation)

            print(f'FINISHED GENERATION {generation}\n')
           
            # plot pareto front
            plot = Scatter(title = "Objective Space", labels="f")
            front=algorithm.pop.get("F")
            plot.add(-front)
            temp_path = os.path.join(root_dir, "saved_data", self.experiment_name,"generation_" + str(generation), "pareto_front.pdf")
            plot.save(temp_path)

            #make directory for next generation
            generation+=1
            create_controller_dir(self.experiment_name,generation,self.is_NSGC)
            
            
def run_multi_ga(
    experiment_name, 
    structure_shape, 
    pop_size, 
    total_generation, 
    train_iters, 
    num_cores,
    env_name1,
    env_name2,
    seed=0,
    is_two_env_parallel=False,
    is_ist=False,
    is_NSGC=False,
    dim_map=2,
    n_niches=128,
    params=cm.default_params,):
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
        
        if is_ist:
            print("FORCE OVERRIDE")
            ans="y"
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
            f'max generation: {total_generation}, train iters {train_iters}.')
        
        f.close()

    archive = {}
    novelty_archive=[]

    c = cm.cvt(n_niches, dim_map,
               params['cvt_samples'], params['cvt_use_cache'])
    kdt = KDTree(c, leaf_size=30, metric='euclidean')

    #Multiobjective Initialization
    algorithm=NSGA2(
        pop_size=pop_size,
        sampling=MySampling(structure_shape,experiment_name,is_NSGC),
        mutation=MyMutation(),
        crossover=MyCrossover(),
        eliminate_duplicates=False
    )

    res = minimize(MyProblem(algorithm,tc,num_cores,env_name1,env_name2,novelty_archive,archive,kdt,__add_to_archive,experiment_name,is_two_env_parallel,is_NSGC),
                algorithm,
                ('n_gen',total_generation),
                seed=seed,
                callback=MyCallback(experiment_name,is_NSGC,archive,n_niches),
                verbose=False)

    plot = Scatter(title = "Objective Space", labels="f")
    plot.add(-res.F)
    temp_path = os.path.join(root_dir, "saved_data", experiment_name, "pareto_front_final.pdf")
    plot.save(temp_path)

        