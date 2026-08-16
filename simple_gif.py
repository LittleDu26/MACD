import os
root_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch
import gym
from utils.algo_utils import *
from macd.envs import make_vec_envs
import macd.helper as helper
import evogym.envs
import imageio
from pygifsicle import optimize


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")

def save_robot_gif(method,env_name,seed,id):
    save_path_controller = os.path.join(root_dir,'result',method,env_name,str(seed),'controllers',str(id)+'.pt')
    save_path_structure = os.path.join(root_dir,'result',method,env_name,str(seed),'structures',str(id)+'.npz')
    structure_data = np.load(save_path_structure)
    structure = []
    for key, value in structure_data.items():
        structure.append(value)
    structure = tuple(structure)

    robot = Structure(*structure, 0)
    print(robot)

    env = make_vec_envs(env_name, [robot], seed=101, gamma=None, device=device, ret=False, ob=True)
                    
    uni_agent, obs_rms = torch.load(save_path_controller, map_location=device)
    uni_agent.eval()
    uni_agent.to(device)
    uni_agent.ac.device=device

    vec_norm = helper.get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = obs_rms

    obs = env.reset()
    obs['stage'] = torch.from_numpy(np.array([1.0]))
    imgs = []
    eval_episode_rewards = []
    # Rollout
    while True:
        with torch.no_grad():
            val, action, logp,  = uni_agent.uni_act(obs, mean_action=True)
        obs, reward, done, infos = env.step(action)
        obs['stage'] = torch.from_numpy(np.array([1.0]))
        # img = env.render(mode='img')  
        # imgs.append(img)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])
        # Done
        if len(eval_episode_rewards)==1:
            break
    env.close()
    print("Evalution done!")
    # gif_path = os.path.join(root_dir,'visual',f'{env_name}-{id}-{eval_episode_rewards[0]}.gif')
    # imageio.mimsave(gif_path, imgs, duration=(1/50.0))
    # print("GIF save to : ", gif_path)
    # try:
    #     optimize(gif_path)
    # except:
    #     print("Error optimizing gif. Most likely cause is that gifsicle is not installed.")
    #     print("Please run this command: sudo apt-get install -y gifsicle ")
    return eval_episode_rewards[0]

if __name__ == '__main__':

    id_list=[37]
    method='MACD_noS'
    seed=0
    env_name = 'Walker-v0'
    for id in id_list:
        print("id:"+str(id)+"\t",save_robot_gif(method,env_name,seed,id))