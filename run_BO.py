from utils.MyUtils import get_par
from bo.run import run_bo
import os
import random
import numpy as np
root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':

    tasks=["Walker-v0"]
    seed=101
    random.seed(seed)
    np.random.seed(seed)
    num=1
    for i in range(len(tasks)):
        env_name = tasks[i]
        max_eva, tc = get_par(env_name)
        for i in range(num):
            save_path = os.path.join(root_dir, 'result','BO',env_name,f"{i}")        
            run_bo(
                env_name=env_name,
                structure_shape=(5, 5),
                pop_size= 20,
                max_evaluations=max_eva,
                train_iters=tc,
                num_cores=20,
                device_num=0,
                save_path=save_path,
                seed=seed
                )