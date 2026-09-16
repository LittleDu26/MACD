import sys,os,math
from ppo.envs import make_vec_envs
from ppo.utils import get_vec_normalize
import torch
import imageio
from pygifsicle import optimize
import numpy as np

GIF_RESOLUTION = (1280/5, 720/5)

def make_id_centroid_dict(exp_name, load_dir, gen,use_cells):
    if use_cells!=True:
        return None
    id_centroid_dict={}
    gen_data_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "centroid_score.txt")
    with open(gen_data_path,'r') as f:
        for line in f:
            id_centroid_dict[int(line.split()[0])]=(float(line.split()[1]),float(line.split()[2]))
    return id_centroid_dict

def id_to_centroid(id_centroid_dict,use_cells,idx):
    cent_x=None
    cent_y=None
    if use_cells:
        try:
            cent_x,cent_y=id_centroid_dict[idx]
        except:
            print("NoKeyFoundError: id is not found in centroid_score folder or dictionary from it.")
    return cent_x,cent_y

def save_robot_gif(out_path, env_name, body_path, ctrl_path,cent=(None,None)):
    global GIF_RESOLUTION

    structure_data = np.load(body_path)
    structure = []
    for key, value in structure_data.items():
        structure.append(value)
    structure = tuple(structure)
    
    env = make_vec_envs(env_name, structure, 1000, 1, None, None, device='cpu', allow_early_resets=False)
    env.get_attr("default_viewer", indices=None)[0].set_resolution(GIF_RESOLUTION)
                    
    actor_critic, obs_rms = torch.load(ctrl_path, map_location='cpu')
    
    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms
    
    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    
    obs = env.reset()
    #死にポイント
    img = env.render(mode='img')
    reward = None
    done = False
    
    
    imgs = []
    # arrays = []
    while not done:

        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)

        obs, reward, done, _ = env.step(action)
        img = env.render(mode='img')
        imgs.append(img)

        masks.fill_(0.0 if (done) else 1.0)

        if done == True:
            env.reset()

    env.close()
    
    imageio.mimsave(f'{out_path}.gif', imgs, duration=(1/50.0))
    try:
        optimize(out_path)
    except:
        pass
        # print("Error optimizing gif. Most likely cause is that gifsicle is not installed.")
    return 0

def get_exp_gen_data(exp_name, load_dir, gen,is_multi=False):
    robot_data = []
    gen_data_path = os.path.join(load_dir, exp_name, f"generation_{gen}", "output.txt")
    f = open(gen_data_path, "r")
    for line in f:
        line_content=line.split()
        try:
            _=int(line_content[0])
            if is_multi:
                robot_data.append((int(line_content[0]), float(line_content[1]),float(line_content[2])))
            else:
                robot_data.append((int(line_content[0]), float(line_content[1]))) 
        except:
            pass
              
    return robot_data


def load_dir_calc(pop_size,idx,use_cells,gen):
    if use_cells:
        return math.floor(idx/pop_size)
    else:
        return gen
    

# automatically get generation list from folder
def get_generations(load_dir, exp_name):
    gen_list = os.listdir(os.path.join(load_dir, exp_name))
    gen_count = 0
    while gen_count < len(gen_list):
        try:
            gen_list[gen_count] = int(gen_list[gen_count].split("_")[1])
        except:
            del gen_list[gen_count]
            gen_count -= 1
        gen_count += 1
    return [i for i in range(gen_count)]


class Robot():
    def __init__(
        self, 
        body_path=None, 
        ctrl_path=None, 
        reward=None, 
        env_name=None, 
        exp_name=None, 
        idx=None,
        gen=None,
        cent=None
        ):
        self.body_path = body_path
        self.ctrl_path = ctrl_path
        self.reward = reward
        self.env_name = env_name
        self.exp_name = exp_name
        self.gen = gen
        self.idx=idx
        self.cent=cent
    def __str__(self):
        exp_str = f'{self.exp_name}' if self.exp_name is not None else ''
        gen_str = f'gen{self.gen}' if self.gen is not None else ''
        reward_str = f'({round(self.reward, 3)})' if self.reward is not None else ''
        comps = [exp_str, gen_str, reward_str]
        out = ''
        for comp in comps:
            if len(comp) != 0:
                out += f'{comp}_'
        return out[:-1]
