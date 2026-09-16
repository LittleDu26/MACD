from ast import arg
from enum import Flag
import random
import numpy as np
import torch
import math

from ga.run import run_ga
from ga.run_tournament import run_ga_tournament
from ppo.arguments import get_args
from utils.algo_utils import calc_GAEval_from_SHEvaluations
from utils.macd_params import get_par

def main():
    args=get_args()
    seed = args.randseed
    random.seed(seed)
    np.random.seed(seed)

    train_scale=1

    env_max_eval=\
    {
        'Walker-v0':500,
        'BridgeWalker-v0':1000,
        "UpStepper-v0":3000//train_scale,
        "PlatformJumper-v0":6000,
        'ObstacleTraverser-v0':3000,
        'ObstacleTraverser-v1':6000,
        'Hurdler-v0':6000,
        'GapJumper-v0':6000,
        'WingspanMazimizer-v0':2000,
        'Lifter-v0':6000,
        'CaveCrawler-v0':3000,
        'Carrier-v0':1500,
        'Catcher-v0':6000,
        'Balancer-v0':2000,
    }

    env_name=None
    env_name='BridgeWalker-v0'
    #env_name='Walker-v0'
    #env_name="UpStepper-v0"
    #env_name="PlatformJumper-v0"
    #env_name="ObstacleTraverser-v1"
    #env_name='ObstacleTraverser-v0'
    #env_name='Hurdler-v0'
    #env_name='GapJumper-v0'
    #env_name='WingspanMazimizer-v0'
    #env_name='Lifter-v0'
    #env_name='CaveCrawler-v0'
    #env_name='Carrier-v0'
    #env_name='Catcher-v0'
    #env_name='Balancer-v0'

    env_name=args.env_name_for_ist if args.env_name_for_ist!=None else env_name

    is_pruning=True if args.is_pruning==1 else False
    is_ist=args.resume_gen is not None
    resume_gen=args.resume_gen
    robot_size=5
    scale=1
    ga_num_cores = args.ga_num_cores if args.ga_num_cores is not None else 20

    is_tournament=False
    is_transfer=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''
    suffix="scale"+str(scale)+"_" if scale > 1 else ""
    suffix+="SuHa" if is_pruning else ""
    suffix+="train_scale"+str(train_scale)+"_" if train_scale > 1 else ""
    suffix+=str(robot_size)+'*'+str(robot_size) if robot_size!=5 else ''
    
    macd_max_evaluations, macd_train_iters = get_par(env_name)
    train_iters=macd_train_iters if macd_train_iters > 0 else 1024*train_scale
    eval_timing_arr=[64*train_scale,128*train_scale,256*train_scale,512*train_scale]

    #train_iters=1024
    #eval_timing_arr=[64,128,256,512]
    #train_iters=128
    #eval_timing_arr=[8,16,32,64]
    max_evaluations=macd_max_evaluations if macd_max_evaluations > 0 else env_max_eval.get(env_name, 0)
    if max_evaluations == 0:
        raise ValueError("Unknown environment: {}".format(env_name))

    assert args.eval_interval is not None
    for eval_timing in eval_timing_arr:
        assert eval_timing%args.eval_interval==0
    assert is_ist==False or resume_gen!=None

    



    if is_tournament:
        run_ga_tournament(
            experiment_name = env_name+"_Tournament"+suffix+"GA_seed:"+str(seed),
            structure_shape = (5,5),
            pop_size = 32,
            train_iters = train_iters,
            num_cores = 8,
            env_name=env_name,
            max_evaluations = 1024,
            eval_timing_arr=eval_timing_arr,
            is_pruning=is_pruning,
            scale=scale,
            is_ist=is_ist,
            resume_gen=resume_gen,
        )
    else:
        run_ga(
            experiment_name = env_name+"_"+suffix+"GA"+'_seed:'+str(seed),
            structure_shape = (robot_size,robot_size),
            pop_size = 20*scale,
            train_iters = train_iters,
            num_cores = ga_num_cores if is_pruning or is_ist else 4,
            env_name=env_name,
            max_evaluations = max_evaluations if is_pruning else math.ceil(calc_GAEval_from_SHEvaluations(max_evaluations)),
            eval_timing_arr=eval_timing_arr,
            is_pruning=is_pruning,
            scale=scale,
            is_ist=is_ist,
            resume_gen=resume_gen,
            stop_by_train_budget=args.budget_termination == 1,
            train_iter_budget=max_evaluations * train_iters,
            eval_interval=args.eval_interval,
        )


if __name__ == "__main__":
    main()
