import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'PyTorch-NEAT'))
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))
EXPERIMENT_PARENT_DIR = os.path.join(root_dir, 'visual')
import argparse
import sys
from pygifsicle import optimize
import numpy as np
import torch
import gym
import imageio
import pandas as pd
from utils.algo_utils import *
from ppo.envs import make_vec_envs
from ppo.utils import get_vec_normalize
import evogym.envs

def visualize_group_ppo(args, exp_name,id):

    save_path_structure = os.path.join(EXPERIMENT_PARENT_DIR,f'{id}.npz')
    save_path_controller = os.path.join(EXPERIMENT_PARENT_DIR,f'robot_{id}_controller.pt')
    structure_data = np.load(save_path_structure)
    structure = []
    for key, value in structure_data.items():
        structure.append(value)
    structure = tuple(structure)

    env = make_vec_envs(
        exp_name,
        structure,
        1,
        1,
        None,
        None,
        device='cpu',
        allow_early_resets=False)
    # We need to use the same statistics for normalization as used in training
    try:
        actor_critic, obs_rms = torch.load(save_path_controller,map_location='cpu')
    except:
        print(f'\nCould not load robot controller data at {save_path_controller}.\n')

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1,actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)

    obs = env.reset()

    total_steps = 0
    reward_sum = 0
    imgs = []
    while True:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=args.det)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)
        masks.fill_(0.0 if (done) else 1.0)
        reward_sum += reward

        # env.render('screen')
        # imgs.append(env.render('img'))

        if done == True:
            env.reset()
            reward_sum = float(reward_sum.numpy().flatten()[0])
            print(f'\ntotal reward: {round(reward_sum, 5)}\n')
            break
        total_steps += 1
        
    env.venv.close()
    # gif_path =os.path.join(EXPERIMENT_PARENT_DIR,exp_name,f"GA_{exp_name}.gif")
    # imageio.mimsave(gif_path, imgs, duration=(1 / 50.0))
    # print("GIF save to : ", gif_path)
    # try:
    #     optimize(gif_path)
    # except:
    #     print("Error optimizing gif. Most likely cause is that gifsicle is not installed.")
    #     print("Please run this command: sudo apt-get install -y gifsicle ")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
        '--env-name',
        help='environment to train on')
    parser.add_argument(
        '--non-det',
        action='store_true',
        default=False,
        help='whether to use a non-deterministic policy')
    args = parser.parse_args()
    args.det = True
    id=8
    exp_name = "Walker-v0"
    visualize_group_ppo(args, exp_name,id)