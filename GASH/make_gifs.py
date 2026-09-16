
import os, sys
root_dir = os.path.dirname(os.path.abspath(__file__))
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))

from utils.algo_utils import *
from gif_utils import *

def dummy_callback(_):
    pass

class Job():
    def __init__(
        self, 
        name, 
        experiment_names, 
        env_names, 
        load_dir,
        generations=None, 
        population_size=None,
        ranks=None,
        jobs=None,
        use_cells=False,
        organize_by_jobs=True,
        organize_by_experiment=False,
        organize_by_generation=False,
        vis_64=None):

        # set values
        self.name = name
        self.experiment_names = experiment_names
        self.env_names = env_names
        self.load_dir = load_dir
        self.generations = generations
        self.population_size = population_size
        self.ranks = ranks
        self.use_cells=use_cells
        self.vis_64=vis_64

        # set jobs 
        self.sub_jobs = []
        if jobs:
            for job in jobs:
                self.sub_jobs.append(job)
                self.sub_jobs[-1].name = job.name if organize_by_jobs else None
        if organize_by_experiment:
            for exp_name, env_name in zip(self.experiment_names, self.env_names):
                self.sub_jobs.append(Job(
                    name = exp_name,
                    experiment_names = [exp_name],
                    env_names = [env_names],
                    load_dir = self.load_dir,
                    generations = self.generations,
                    ranks = self.ranks,
                    organize_by_experiment=False,
                    organize_by_generation=organize_by_generation
                ))
            self.experiment_names = None
            self.env_names = None
            self.generations = None
            self.ranks = None
        elif organize_by_generation:
            assert len(self.experiment_names) == 1, (
                'Cannot create generation level folders for multiple experiments. Quick fix: set organize_by_experiment=True.'
            )
            if self.generations == None:
                exp_name = self.experiment_names[0]
                self.generations = get_generations(self.load_dir, exp_name)
            for gen in self.generations:
                self.sub_jobs.append(Job(
                    name = f'generation_{gen}',
                    experiment_names = self.experiment_names,
                    env_names = self.env_names,
                    load_dir = self.load_dir,
                    generations = [gen],
                    population_size=self.population_size,
                    ranks = self.ranks,
                    use_cells=self.use_cells,
                    organize_by_experiment=False,
                    organize_by_generation=False,
                    vis_64=self.vis_64
                ))
            self.experiment_names = None
            self.env_names = None
            self.generations = None
            self.ranks = None

    def generate(self, load_dir, save_dir, depth=0):
        if self.name is not None and len(self.name) != 0:
            save_dir = os.path.join(save_dir, self.name)

        tabs = '  '*depth
        print(f"{tabs}\{self.name}")
    
        try: os.makedirs(save_dir)
        except: pass

        for sub_job in self.sub_jobs:
            sub_job.generate(load_dir, save_dir, depth+1)

        # collect robots
        if self.experiment_names == None:
            return 

        robots = []
        for exp_name, env_name in zip(self.experiment_names, self.env_names):
            exp_gens = self.generations if self.generations is not None else get_generations(self.load_dir, exp_name)
            for gen in exp_gens:
                id_centroid_dict=make_id_centroid_dict(exp_name, load_dir, gen,self.use_cells)
                for idx, reward in get_exp_gen_data(exp_name, load_dir, gen):
                    cent_x,cent_y=id_to_centroid(id_centroid_dict,self.use_cells,idx)
                    ctrl_dir="controller64" if self.vis_64==True else "controller"
                    robots.append(Robot(
                        body_path = os.path.join(load_dir, exp_name, f"generation_{load_dir_calc(self.population_size,idx,False,gen)}", "structure", f"{idx}.npz"),
                        ctrl_path = os.path.join(load_dir, exp_name, f"generation_{load_dir_calc(self.population_size,idx,False,gen)}", ctrl_dir, f"robot_{idx}_controller.pt"),
                        reward = reward,
                        env_name = env_name,
                        exp_name = exp_name if len(self.experiment_names) != 1 else None,
                        gen = gen if len(exp_gens) != 1 else None,
                        idx=idx,
                        cent=(cent_x,cent_y),
                    ))

        # sort and generate
        robots = sorted(robots, key=lambda x: x.reward, reverse=True)
        ranks = self.ranks if self.ranks is not None else [i for i in range(len(robots))]
        
        # make gifs
        for i, robot in zip(ranks, robots):
            cent_info= '' if robot.cent==(None,None) else 'x='+str(robot.cent[0])+', y='+str(robot.cent[1])
            suffix="_64" if self.vis_64==True else ""
            save_robot_gif(
                os.path.join(save_dir, f'{robot}_{str(robot.idx)}_{cent_info}_{suffix}'),
                robot.env_name,
                robot.body_path,
                robot.ctrl_path
            )
    
# NUM_PROC = 8
if __name__ == '__main__':
    exp_root = os.path.join('saved_data')
    save_dir = os.path.join(root_dir, 'saved_data', 'all_media')
    env_name="PlatformJumper-v0"
    seed=101
    experiment_name = env_name+"_SuHaGA_seed:"+str(seed)
    use_cells=False
    is_transfer=False
    transfer_gen=100
    suffix="_transfer:gen="+str(transfer_gen) if is_transfer else ''

    '''
    env_name1="Jumper-v0"
    env_name2="PlatformJumper-v0"
    is_specialJump=False
    suffix='_forwardJump' if is_specialJump else ''
    load_seed=20
    score=8.52
    expr_name=env_name1+"_"+env_name2+suffix+'_seed:'+str(load_seed)
    '''

    #experiment_name='Repro_from_'+expr_name+'_score:'+str(score)
    my_job = Job(
        #name = experiment_name+suffix+'_seed:'+str(seed),
        name = experiment_name,
        #experiment_names= [experiment_name+suffix+'_seed:'+str(seed)],
        experiment_names=[experiment_name],
        env_names = [env_name],
        load_dir = exp_root,
        generations=[96],
        population_size=32,
        ranks = [i for i in range(27)], #not use when use_cells=True
        use_cells=use_cells,
        organize_by_experiment=False,
        organize_by_generation=True,
        vis_64=False,
    )
    print(save_dir,experiment_name)
    
    my_job.generate(load_dir=exp_root, save_dir=save_dir)