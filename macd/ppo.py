import torch
import torch.optim as optim
import torch.nn as nn
import copy
from .evaluate import evaluate
from .envs import make_vec_envs
from . import helper
from utils.algo_utils import Structure
from .transformer.buffer import Buffer
import gym
import os

# PPO class
class PPO:
    def __init__(self, robot=None, train_iters=None, ppo_size=None, agent=None,
                 verbose=True, ppo_args=None, save_path=None, total_step=None,
                 device=None,mmse=False):
        self.args = ppo_args
        self.verbose = verbose
        self.agent = agent
        self.ppo_size = ppo_size
        self.robots = []
        self.train_iters = train_iters
        self.robots_tuple = robot
        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.lr, eps=self.args.EPS)
        self.robots = []
        self.mmse = mmse
        self._evaluation_records = []
        for i in range(self.ppo_size):
            self.robots.append(Structure(*robot, i))
        if self.mmse:
            self.total_step = self.train_iters
        else:
            self.total_step = total_step
            
        self.device =device
        # Logger
        self.save_to = save_path

        # Set replay buffer
        self.buffer = self.reset_buffer()
        self.buffer.to(self.device)

    @property
    def evaluation_records(self):
        """Evaluation trajectory collected during ``train``.

        A tuple of copied dictionaries is returned so callers can inspect the
        learning curve without mutating PPO's internal record.
        """
        return tuple(dict(record) for record in self._evaluation_records)

    def reset_buffer(self):
        train_env = gym.make(self.args.env_name, mode='modular', body=self.robots_tuple[0],
                             connections=self.robots_tuple[1], env_id=self.args.env_name)
        sequence_size = train_env.voxel_num
        obs_sample = train_env.reset()
        train_env.close()
        return Buffer(obs_sample, act_shape=sequence_size, num_envs=self.ppo_size, cfg=self.args)

    def train(self,id,iteration,start_fitness):
        self._evaluation_records = []
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
        # Make envs
        self.envs = make_vec_envs(self.args.env_name, self.robots, self.args.seed,
                                  self.args.GAMMA, self.device, ob=True, ret=True)
        initial_fitness=None
        if iteration==0:
            obs_rmss = helper.get_vec_normalize(self.envs).ob_rms
            initial_fitness = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent, ob_rms=obs_rmss,
                                       env_name=self.args.env_name, init_robots=[self.robots[0]], seed=self.args.seed,
                                       device=self.device)
            curve_best_fitness = float(initial_fitness)
            self._evaluation_records.append({
                "update": 0,
                "env_steps": 0,
                "return": float(initial_fitness),
                "best_return": curve_best_fitness,
            })
        else:
            curve_best_fitness = float(start_fitness)
        obs = self.envs.reset()
        num_updates = int(self.args.num_env_steps) // self.args.TIMESTEPS

        reward_history=[]
        max_fitness = start_fitness
        for i,j in enumerate(range(num_updates*4)[iteration:]):
            if self.args.use_linear_lr_decay:
                # decrease learning rate linearly
                helper.update_linear_schedule(self.optimizer, j, num_updates*4,self.args.lr)
            # print("Collect experience !!!!!")
            for step in range(self.args.TIMESTEPS):
                # Sample actions
                val, act, logp= self.agent.uni_act(obs)
                next_obs, reward, done, infos = self.envs.step(act)
                masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device,
                )
                timeouts = torch.tensor(
                    [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.buffer.insert(obs, act, logp, val, reward, masks, timeouts)
                obs = next_obs

            next_val = self.agent.get_value(obs)
            self.buffer.compute_returns(next_val)

            # print("Begin training!!!!!")
            self.train_on_batch()

            if (i+1) % self.args.eval_interval == 0:
                # Evaluation
                obs_rmss = helper.get_vec_normalize(self.envs).ob_rms
                fitness = evaluate(num_evals=self.args.num_evals, uni_agent=self.agent, ob_rms=obs_rmss,
                                   env_name=self.args.env_name,init_robots=[self.robots[0]], seed=self.args.seed,
                                   device=self.device)
                curve_best_fitness = max(curve_best_fitness, float(fitness))
                self._evaluation_records.append({
                    "update": int(j + 1),
                    "env_steps": int((j + 1) * self.args.TIMESTEPS),
                    "return": float(fitness),
                    "best_return": curve_best_fitness,
                })
                if fitness>max_fitness:
                    max_fitness=fitness
                    self.agent.obs_rms = copy.deepcopy(
                        getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None))
                    temp_path_controller = os.path.join(self.save_to, "controllers", f'{id}.pt')
                    torch.save([self.agent, getattr(helper.get_vec_normalize(self.envs), 'ob_rms', None)],temp_path_controller)
            if (i+1)%self.train_iters==0:
                reward_history.append(max_fitness)
            if (i+1)>=self.total_step:
                self.envs.close()
                return reward_history,j+1,initial_fitness

    def train_on_batch(self):

        adv = self.buffer.ret - self.buffer.val
        adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        for ep in range(self.args.EPOCHS):
            batch_sampler = self.buffer.get_sampler(adv)
            for batch in batch_sampler:
                # Reshape to do in a single forward pass for all steps
                val, logp, ent = self.agent(batch["obs"], batch["act"])
                clip_ratio = self.args.CLIP_EPS
                ratio = torch.exp(logp - batch["logp_old"])

                surr1 = ratio * batch["adv"]
                surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * batch["adv"]

                pi_loss = -torch.min(surr1, surr2).mean()

                if self.args.USE_CLIP_VALUE_FUNC:
                    val_pred_clip = batch["val"] + (val - batch["val"]).clamp(
                        -clip_ratio, clip_ratio
                    )
                    val_loss = (val - batch["ret"]).pow(2)
                    val_loss_clip = (val_pred_clip - batch["ret"]).pow(2)
                    val_loss = 0.5 * torch.max(val_loss, val_loss_clip).mean()
                else:
                    val_loss = 0.5 * (batch["ret"] - val).pow(2).mean()

                self.optimizer.zero_grad()
                loss = val_loss * self.args.VALUE_COEF + pi_loss - ent * self.args.ENTROPY_COEF
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)

                self.optimizer.step()
