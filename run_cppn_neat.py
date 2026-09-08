import os
from utils.MyUtils import get_par
from cppn_neat.run import run_cppn_neat
import random
import numpy as np
root_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    seed = 101
    tasks=["Walker-v0"]
    num=1
    random.seed(seed)
    np.random.seed(seed)
    for i in range(len(tasks)):
        env_name = tasks[i]
        max_eva, tc = get_par(env_name)
        for i in range(num):
            save_path = os.path.join(root_dir, "result","CPPN",env_name,f"{i}")
            best_robot, best_fitness = run_cppn_neat(
                experiment_name=tasks[i],
                structure_shape=(5, 5),
                pop_size=20,
                max_evaluations=max_eva,
                train_iters=tc,
                num_cores=20,
            )
            print('Best robot:')
            print(best_robot)
            print('Best fitness:', best_fitness)