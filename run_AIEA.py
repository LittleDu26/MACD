from utils.MyUtils import get_par
from SAEA.AIEA import run_aiea
import os
import random
import numpy as np
root_dir = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":

    tasks=["Walker-v0"]
    num=1
    seed=101
    random.seed(seed)
    np.random.seed(seed)
    for i in range(len(tasks)):
        env_name = tasks[i]
        max_eva, tc = get_par(env_name)
        for i in range(num):
            save_path = os.path.join(root_dir, "result","AIEA",env_name,f"{i}")
            run_aiea(
                pop_size=20,
                structure_shape=(5, 5),
                max_evaluations=max_eva,
                train_iters=tc,
                num_cores=20,
                env_name=env_name,
                save_path=save_path,
                seed=seed
            )
