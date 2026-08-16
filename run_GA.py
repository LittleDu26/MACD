from ga.run import run_ga
from utils.MyUtils import get_par
import os
root_dir = os.path.dirname(os.path.abspath(__file__))


if __name__ == "__main__":
   
    tasks=["Walker-v0"]
    seed=101
    num=1
    for i in range(len(tasks)):
        env_name = tasks[i]
        max_eva, tc = get_par(env_name)
        for i in range(num):
            save_path = os.path.join(root_dir, 'result','GA',env_name,f"{i}")        
            run_ga(
                pop_size = 20,
                structure_shape = (5,5),
                experiment_name = tasks[i],
                max_ganeration=200,
                max_evaluations=max_eva,
                train_iters = tc,
                num_cores=20,
                device_num=0,
                save_path=save_path,
                seed=seed
                )