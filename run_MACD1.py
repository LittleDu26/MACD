import argparse
from macd.run import run
import os
import json
from utils.MyUtils import get_par
root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, default='Walker-v0',
                        help='env')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--target_size', type=int, default=5,
                        help='target design space, default 5x5')
    parser.add_argument('--threads_num', type=int, default=20,
                        help='number of CPU threads')
    parser.add_argument('--max_iters', type=int, default=50000,
                        help='')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='')
    parser.add_argument('--train_iters', type=int, default=64,
                        help='')
    parser.add_argument('--total_step', type=int, default=1000,
                        help='')
    parser.add_argument('--save_to', type=str, default='',
                        help='save_to')
    parser.add_argument('--suffix', type=str, default='',
                        help='experiment name suffix, e.g. XX -> MACD(XX)')
    parser.add_argument('--distill', action='store_true',
                        help='')
    parser.add_argument('--mmse', action='store_true',
                        help='')
    parser.add_argument('--promotion_k', type=int, default=10,
                        help='maximum active morphologies promoted each generation')
    parser.add_argument('--lambda_max', type=float, default=0.30,
                        help='initial weight for signed fitness improvement in survivor selection')
    parser.add_argument('--lambda_min', type=float, default=0.05,
                        help='late-stage weight for signed fitness improvement in survivor selection')
    parser.add_argument('--lambda_tau', type=float, default=2.0,
                        help='decay constant for signed improvement weight')
    parser.add_argument('--selection_eps', type=float, default=1e-8,
                        help='numerical epsilon for survivor selection normalization')

    args = parser.parse_args()
    args.seed = 101
    args.target_size = 5
    args.threads_num = 20
    args.pop_size = 20
    args.train_iters=64
    args.mmse = True
    args.distill = True
    args.suffix=''
    # env_list =["PlatformJumper-v0",
    #             "BridgeWalker-v0",
    #             "DownStepper-v0",
    #             "Hurdler-v0"]
    env_list = ["Carrier-v0","Thrower-v0"]
    num=1

    for env in env_list:
        max_eva, tc = get_par(env)
        args.env=env
        args.total_step=tc
        args.max_iters=max_eva*tc
        temp_save_to=os.path.join(root_dir, "result1","MACD")
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
