import argparse
from macd.run import run
import os
import json
import sys
from utils.MyUtils import get_par
root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, default='Walker-v0',
                        help='env name, or all for the nine paper tasks')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--target_size', type=int, default=5,
                        help='target design space, default 5x5')
    parser.add_argument('--threads_num', type=int, default=20,
                        help='number of CPU threads')
    parser.add_argument('--max_iters', type=int, default=None,
                        help='global PPO-update budget; default: task budget')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='')
    parser.add_argument('--train_iters', type=int, default=64,
                        help='')
    parser.add_argument('--total_step', type=int, default=None,
                        help='maximum PPO updates per robot; default: task setting')
    parser.add_argument('--save_to', type=str, default='',
                        help='save_to')
    parser.add_argument('--suffix', type=str, default='',
                        help='experiment name suffix, e.g. XX -> MACD(XX)')
    parser.add_argument('--distill', action='store_true', default=True,
                        help='')
    parser.add_argument('--mmse', action='store_true', default=True,
                        help='')
    parser.add_argument('--promotion_k', type=int, default=10,
                        help='maximum active morphologies promoted each generation')
    parser.add_argument('--lambda_max', type=float, default=0.30,
                        help='initial weight for signed fitness improvement in survivor selection')
    parser.add_argument('--lambda_min', type=float, default=0.05,
                        help='late-stage weight for signed improvement in survivor selection')
    parser.add_argument('--lambda_tau', type=float, default=2.0,
                        help='decay constant for signed improvement weight')
    parser.add_argument('--selection_eps', type=float, default=1e-8,
                        help='numerical epsilon for survivor selection normalization')
    parser.add_argument('--check', action='store_true',
                        help='check the selected environments without training')

    args = parser.parse_args()
    paper_env_list = ["Walker-v0","AreaMaximizer-v0","Carrier-v0",
                      "Thrower-v0","UpStepper-v0","ObstacleTraverser-v0",
                      "ObstacleTraverser-v1","GapJumper-v0","BeamSlider-v0"]
    env_list = paper_env_list if args.env == 'all' else [args.env]

    if args.check:
        import gym
        import torch
        import evogym.envs
        from evogym import sample_robot, get_full_connectivity

        body, _ = sample_robot((args.target_size, args.target_size))
        for env in env_list:
            max_eva, tc = get_par(env)
            check_env = gym.make(env, mode='modular', body=body,
                                 connections=get_full_connectivity(body), env_id=env)
            check_env.reset()
            check_env.close()
            print(f"OK {env}: evaluations={max_eva}, max_updates_per_robot={tc}")
        print(f"Environment check passed (PyTorch {torch.__version__}, CPU mode).")
        sys.exit(0)

    requested_total_step = args.total_step
    requested_max_iters = args.max_iters
    requested_save_to = args.save_to
    num=1

    for env in env_list:
        max_eva, tc = get_par(env)
        args.env=env
        args.total_step=tc if requested_total_step is None else requested_total_step
        args.max_iters=max_eva*args.total_step if requested_max_iters is None else requested_max_iters
        temp_save_to=requested_save_to if requested_save_to else os.path.join(root_dir, "result","MACD")
        if not args.mmse:
            temp_save_to += "_noM"
        if not args.distill:
            temp_save_to += "_noS"
        if args.suffix:
            temp_save_to += f"({args.suffix})"
        for i in range(num):
            args.save_to=os.path.join(temp_save_to,args.env,str(i))
            if not os.path.exists(args.save_to):
                os.makedirs(args.save_to)
                args_file_path = os.path.join(args.save_to, "config.json")
                with open(args_file_path, "w") as f:
                    json.dump(vars(args), f, indent=4)
            else:
                continue
            run(args)
