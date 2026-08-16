from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

from .daab_model import DAABActor, DAABCritic, check_finite_tensor
from .distributions import DiagGaussian, FixedDiagGaussian
from .transformermodel import TransformerModel


class TransformerPPOAC(nn.Module):
    def __init__(
        self,
        modular_state_dim,
        modular_action_dim,
        sequence_size,
        other_feature_size,
        ppo_args=None,
        trans_args=None,
        ac_type=None,
        controller_type=None,
        device=None,
    ):
        super(TransformerPPOAC, self).__init__()
        self.sequence_size = sequence_size
        self.other_feature_size = other_feature_size
        self.state_dim = modular_state_dim
        self.action_dim = modular_action_dim
        self.ppo_args = ppo_args
        self.trans_args = trans_args
        self.ac_type = ac_type
        self.controller_type = controller_type or getattr(trans_args, "controller_type", "original")
        self.device = device

        if self.controller_type == "daab":
            self.v_net = DAABCritic(sequence_size, other_feature_size, trans_args)
            self.mu_net = DAABActor(sequence_size, other_feature_size, trans_args)
        elif self.controller_type == "original":
            common = dict(
                feature_size=self.state_dim,
                output_size=self.action_dim,
                sequence_size=self.sequence_size,
                other_feature_size=self.other_feature_size,
                ninp=trans_args.attention_embedding_size,
                nhead=trans_args.attention_heads,
                nhid=trans_args.attention_hidden_size,
                nlayers=trans_args.attention_layers,
                dropout=trans_args.dropout_rate,
                args=trans_args,
                use_transformer=self.ac_type,
            )
            self.v_net = TransformerModel(is_actor=False, **common)
            self.mu_net = TransformerModel(is_actor=True, **common)
        else:
            raise ValueError("Unknown controller_type: {}".format(self.controller_type))

        self.num_actions = self.sequence_size
        if self.ppo_args.ACTION_STD_FIXED:
            self.act_dist = FixedDiagGaussian(self.num_actions, ppo_args.ACTION_STD)
        else:
            self.act_dist = DiagGaussian(self.num_actions)

    def forward(self, state, act=None):
        modular_state = state["modular"]
        other_state = state["other"]
        act_mask = state["act_mask"].bool()
        obs_padding = state["obs_mask"].bool()
        batch_size = modular_state.shape[0]

        if self.controller_type == "daab":
            daab_state = state["daab_modular"].reshape(batch_size, self.sequence_size, 12)
            voxel_type = state["voxel_type"].reshape(batch_size, self.sequence_size)
            grid_pos = state["grid_pos"].reshape(batch_size, self.sequence_size, 2)
            token_index = state["token_index"].reshape(batch_size, self.sequence_size)
            val, _ = self.v_net(
                daab_state, other_state, voxel_type, grid_pos, obs_padding
            )
            mu, _ = self.mu_net(
                daab_state, other_state, voxel_type, grid_pos, token_index, obs_padding
            )
        else:
            input_state = modular_state.reshape(batch_size, self.sequence_size, -1).permute(1, 0, 2)
            obs_coord = self.make_grid_coords(5, device=modular_state.device)
            module_vals = self.v_net(input_state, other_state, obs_padding, obs_coord)
            module_vals = module_vals * (1 - obs_padding.int())
            num_limbs = self.sequence_size - torch.sum(obs_padding.int(), dim=1, keepdim=True)
            val = torch.sum(module_vals, dim=1, keepdim=True) / num_limbs.clamp(min=1)
            mu = self.mu_net(input_state, other_state, obs_padding, obs_coord)

        check_finite_tensor("value", val, self.controller_type)
        check_finite_tensor("actor_mean", mu, self.controller_type)
        pi = self.act_dist(mu)
        if act is None:
            return val, pi

        logp = pi.log_prob(act)
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        entropy = pi.entropy()
        entropy[act_mask] = 0.0
        entropy = entropy.sum(-1, keepdim=True)
        return val, pi, logp, entropy

    def reset_seq_size(self, seq_size):
        self.sequence_size = seq_size
        self.num_actions = seq_size
        self.mu_net.reset_seq_size(seq_size)
        self.v_net.reset_seq_size(seq_size)
        if self.ppo_args.ACTION_STD_FIXED:
            self.act_dist = FixedDiagGaussian(self.num_actions, self.ppo_args.ACTION_STD)
        else:
            self.act_dist = DiagGaussian(self.num_actions)

    def make_grid_coords(self, width, scale=10.0, device=None):
        x = torch.arange(1, width + 1, device=device)
        y = torch.arange(1, width + 1, device=device)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1).float() / scale

    def attention_forward(self, state, net_name="mu", attn_mask=None,
                          return_hidden_states=False):
        batch_size = state["modular"].shape[0]
        obs_padding = state["obs_mask"].bool()
        net = self.mu_net if net_name == "mu" else self.v_net
        if self.controller_type == "daab":
            daab_state = state["daab_modular"].reshape(batch_size, self.sequence_size, 12)
            voxel_type = state["voxel_type"].reshape(batch_size, self.sequence_size)
            grid_pos = state["grid_pos"].reshape(batch_size, self.sequence_size, 2)
            if net_name == "mu":
                token_index = state["token_index"].reshape(batch_size, self.sequence_size)
                return net(
                    daab_state, state["other"], voxel_type, grid_pos,
                    token_index, obs_padding, return_attention=True,
                    return_hidden_states=return_hidden_states,
                )
            return net(
                daab_state, state["other"], voxel_type, grid_pos,
                obs_padding, return_attention=True,
                return_hidden_states=return_hidden_states,
            )

        modular_state = state["modular"].reshape(batch_size, self.sequence_size, -1).permute(1, 0, 2)
        obs_coord = self.make_grid_coords(5, device=modular_state.device)
        return net(
            modular_state, state["other"], obs_padding, obs_coord,
            attn_mask=attn_mask, return_attn=True,
            return_hidden_states=return_hidden_states,
        )


class PPOAgent(nn.Module):
    
    def __init__(self, actor_critic):
        super(PPOAgent, self).__init__()
        self.ac = actor_critic

    def forward(self, obs, act):
        index = obs['stage']
        batch_size = index.shape[0]
        ac_index = np.argwhere(index.cpu().numpy()>0)

        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        ent = torch.zeros(batch_size,1).to(self.ac.device)

        ### ac batch
        if ac_index.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    if ot == 'stage'or ot == 'design':
                        pass
                    else:
                        ac_obs_batch[ot] = ov.view(-1, *ov.size()[1:])[ac_index[:,0]]

            ac_act_batch = act[ac_index[:,0]]
            ac_val, _, ac_logp, ac_ent = self.ac(ac_obs_batch, ac_act_batch)
            
            val[ac_index[:,0]] = ac_val
            logp[ac_index[:,0]] = ac_logp
            ent[ac_index[:,0]] = ac_ent

        ent = ent.mean()
        return val, logp, ent

    @torch.no_grad()
    def uni_act(self, obs, mean_action=False):
        index = obs['stage']
        batch_size = index.shape[0]
        ac_idx = np.argwhere(index.cpu().numpy()>0)
        
        val = torch.zeros(batch_size,1).to(self.ac.device)
        logp = torch.zeros(batch_size,1).to(self.ac.device)
        act = torch.zeros(batch_size,self.ac.sequence_size).to(self.ac.device)

        ### ac batch
        if ac_idx.shape[0]>0:
            if isinstance(obs, dict):
                ac_obs_batch ={}
                for ot, ov in obs.items():
                    if ot == 'stage' or ot == 'design':
                        pass
                    else:
                        ac_obs_batch[ot] = ov[ac_idx[:,0]]
            ac_val, ac_act, ac_logp = self.act(ac_obs_batch, mean_action=mean_action)

            val[ac_idx[:,0]] = ac_val
            logp[ac_idx[:,0]] = ac_logp

            for j in range(ac_idx.shape[0]):
                act[ac_idx[:,0][j]] = ac_act[j]
        return val, act, logp

    @torch.no_grad()
    def act(self, obs, mean_action=False):
        val, pi = self.ac(obs)
        if mean_action:
            act = pi.mode()
        else:
            act = pi.sample()
        logp = pi.log_prob(act)
        act_mask = obs["act_mask"].bool()
        logp[act_mask] = 0.0
        logp = logp.sum(-1, keepdim=True)
        del pi
        return val, act, logp

    @torch.no_grad()
    def get_value(self, obs):
        val, act, logp = self.uni_act(obs)
        return val
