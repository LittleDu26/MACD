import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from evogym import get_full_connectivity
from utils.algo_utils import Structure

from . import helper
from .envs import make_vec_envs
from .transformer.transformerPPOagent import PPOAgent, TransformerPPOAC

DISTILL_LOSS_TYPES = (
    "column_kl",
    "attention_kl",
    "feature_mse",
    "attention_kl_feature_mse",
)
SHARED_ATTENTION_LOSSES = (
    "attention_kl",
    "attention_kl_feature_mse",
)
SHARED_FEATURE_LOSSES = (
    "feature_mse",
    "attention_kl_feature_mse",
)
CONTROLLER_COPY_MODES = ("full_copy", "non_distill_copy")


def same_voxel_mask(parent_robot, child_robot):
    parent_flat = np.asarray(parent_robot).reshape(-1)
    child_flat = np.asarray(child_robot).reshape(-1)
    return (parent_flat == child_flat) & (parent_flat != 0)


def select_attention_parent(child, parent_list):
    best_parent = None
    best_count = -1
    for parent in parent_list:
        if getattr(parent, "best_controller", None) is None:
            continue
        count = int(np.sum(same_voxel_mask(parent.robot, child.robot)))
        if count > best_count:
            best_parent = parent
            best_count = count
    return best_parent


def should_skip_warmup(parent_robot, child_robot):
    same_nonempty_count = int(np.sum(same_voxel_mask(parent_robot, child_robot)))
    return same_nonempty_count < 4


def build_child_controller(parent_controller, sample_setting, ppo_args, trans_args, device):
    actor_critic = TransformerPPOAC(
        modular_state_dim=sample_setting[0],
        modular_action_dim=sample_setting[1],
        sequence_size=sample_setting[3],
        other_feature_size=sample_setting[2],
        ppo_args=ppo_args,
        trans_args=trans_args,
        ac_type="transformer",
        controller_type=getattr(trans_args, "controller_type", "original"),
        device=device,
    )
    child = PPOAgent(actor_critic=actor_critic).to(device)
    initial_state = copy.deepcopy(child.state_dict())
    child.load_state_dict(copy.deepcopy(parent_controller.state_dict()))
    child.to(device)
    _restore_qk_from_initial(child, initial_state)
    return child


def build_copied_child_controller(parent_controller, sample_setting, ppo_args,
                                  trans_args, device, copy_mode):
    """Build a child with either complete or non-distillation inheritance.

    ``non_distill_copy`` preserves the child's random Q/K projections and
    LayerNorm parameters while copying every other compatible parent value.
    For PyTorch's combined QKV projections, only the Q/K slices stay random.
    """
    if copy_mode not in CONTROLLER_COPY_MODES:
        raise ValueError(
            "Unknown controller copy mode {!r}; expected one of {}".format(
                copy_mode, sorted(CONTROLLER_COPY_MODES)
            )
        )

    actor_critic = TransformerPPOAC(
        modular_state_dim=sample_setting[0],
        modular_action_dim=sample_setting[1],
        sequence_size=sample_setting[3],
        other_feature_size=sample_setting[2],
        ppo_args=ppo_args,
        trans_args=trans_args,
        ac_type="transformer",
        controller_type=getattr(trans_args, "controller_type", "original"),
        device=device,
    )
    child = PPOAgent(actor_critic=actor_critic).to(device)
    initial_state = copy.deepcopy(child.state_dict())
    child.load_state_dict(copy.deepcopy(parent_controller.state_dict()))
    child.to(device)
    if copy_mode == "non_distill_copy":
        _restore_distillation_parameters_from_initial(child, initial_state)
    return child


def _restore_qk_from_initial(child, initial_state):
    state = child.state_dict()
    for name, value in state.items():
        if name.endswith("self_attn.in_proj_weight"):
            embed_dim = value.shape[0] // 3
            value[: 2 * embed_dim].copy_(initial_state[name][: 2 * embed_dim])
        elif name.endswith("self_attn.in_proj_bias"):
            embed_dim = value.shape[0] // 3
            value[: 2 * embed_dim].copy_(initial_state[name][: 2 * embed_dim])
        elif name.endswith("self_attn.q_proj.weight") or name.endswith("self_attn.q_proj.bias"):
            value.copy_(initial_state[name])
        elif name.endswith("self_attn.k_proj.weight") or name.endswith("self_attn.k_proj.bias"):
            value.copy_(initial_state[name])


def _restore_distillation_parameters_from_initial(child, initial_state):
    state = child.state_dict()
    for name, value in state.items():
        if name.endswith("self_attn.in_proj_weight") or name.endswith(
            "self_attn.in_proj_bias"
        ):
            embed_dim = value.shape[0] // 3
            value[: 2 * embed_dim].copy_(initial_state[name][: 2 * embed_dim])
        elif _is_separate_qk_parameter(name) or _is_layernorm_parameter(name):
            value.copy_(initial_state[name])


def _is_separate_qk_parameter(name):
    return any(
        name.endswith(suffix)
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.k_proj.bias",
        )
    )


def _is_layernorm_parameter(name):
    # Keep this aligned with _enable_qk_layernorm below.
    return "norm" in name


def collect_parent_rollout_obs(parent_controller, parent_robot, env_name, seed, device,
                               ob_rms=None):
    robot = Structure(parent_robot, get_full_connectivity(parent_robot), 0)
    envs = make_vec_envs(env_name, [robot], seed, None, device, ret=False, ob=True)
    vec_norm = helper.get_vec_normalize(envs)
    if vec_norm is not None and ob_rms is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    parent_controller.eval()
    observations = []
    obs = envs.reset()
    while True:
        if _has_active_stage(obs):
            observations.append(_detach_obs(obs))
        with torch.no_grad():
            _, action, _ = parent_controller.uni_act(obs, mean_action=True)
        obs, _, _, infos = envs.step(action)
        if any("episode" in info for info in infos):
            break

    envs.close()
    return observations


def attention_distill_warmup(parent_controller, child_controller, observations,
                             parent_robot, child_robot, device,
                             batch_size=64, warmup_epochs=2, lr_warm=5e-4,
                             gradient_clip=0.5,
                             loss_type="attention_kl_feature_mse",
                             lambda_attention=1.0, lambda_feature=1.0,
                             lambda_col=1.0):
    _validate_distillation_config(loss_type, lambda_attention, lambda_feature)
    if not observations or should_skip_warmup(parent_robot, child_robot):
        return

    parent_controller.eval()  # 父模型切换到评估模式。
    child_controller.train()  # 子模型切换到训练模式。
    for param in parent_controller.parameters():  # 遍历父模型参数。
        param.requires_grad_(False)  # 关闭父模型梯度。
    for param in child_controller.parameters():  # 遍历子模型参数。
        param.requires_grad_(False)  # 默认关闭子模型梯度。

    trainable, gradient_hooks = _enable_qk_layernorm(child_controller)  # 仅开启指定参数训练。
    optimizer = optim.Adam(trainable, lr=lr_warm)  # 配置 warmup 优化器。

    for _ in range(warmup_epochs):  # 迭代 warmup 轮次。
        order = np.random.permutation(len(observations))  # 打乱样本顺序。
        for start in range(0, len(order), batch_size):  # 分批迭代。
            batch = _stack_obs([observations[i] for i in order[start:start + batch_size]], device)  # 组装批次。
            parent_batch, child_batch, parent_indices, child_indices = _build_distill_views(
                batch, parent_robot, child_robot, device
            )

            loss = 0.0  # 初始化损失累加器。
            for net_name in ("mu", "v"):  # 对两个头进行蒸馏。
                if loss_type == "column_kl":
                    with torch.no_grad():  # 禁用父模型梯度。
                        _, parent_attn = parent_controller.ac.attention_forward(
                            parent_batch, net_name=net_name
                        )
                        s_parent = _column_importance(parent_attn, parent_indices)
                    _, child_attn = child_controller.ac.attention_forward(
                        child_batch, net_name=net_name
                    )
                    s_child = _column_importance(child_attn, child_indices)
                    loss = loss + lambda_col * _kl_divergence(
                        s_parent.detach(), s_child
                    )
                else:
                    with torch.no_grad():
                        _, parent_attn, parent_hidden = (
                            parent_controller.ac.attention_forward(
                                parent_batch,
                                net_name=net_name,
                                return_hidden_states=True,
                            )
                        )
                    _, child_attn, child_hidden = child_controller.ac.attention_forward(
                        child_batch,
                        net_name=net_name,
                        return_hidden_states=True,
                    )
                    loss = loss + _shared_branch_loss(
                        parent_attn,
                        child_attn,
                        parent_hidden,
                        child_hidden,
                        parent_indices,
                        child_indices,
                        loss_type,
                        lambda_attention,
                        lambda_feature,
                    )

            optimizer.zero_grad()  # 清空梯度。
            loss.backward()  # 反向传播。
            nn.utils.clip_grad_norm_(trainable, gradient_clip)  # 梯度裁剪。
            optimizer.step()  # 更新参数。

    for hook in gradient_hooks:
        hook.remove()
    for param in child_controller.parameters():  # 遍历子模型参数。
        param.requires_grad_(True)  # 恢复子模型梯度。


def prepare_distilled_controller(agent, ppo_args, trans_args, sample_setting, args,device):
    parent = agent.distill_parent

    parent_controller = copy.deepcopy(parent.best_controller).to(device)
    child_controller = build_child_controller(
        parent_controller, sample_setting, ppo_args, trans_args, device
    )

    if should_skip_warmup(parent.robot, agent.robot):
        return child_controller

    observations = collect_parent_rollout_obs(
        parent_controller,
        parent.robot,
        args.env,
        args.seed,
        device,
        ob_rms=getattr(parent_controller, "obs_rms", None),
    )

    attention_distill_warmup(
        parent_controller,
        child_controller,
        observations,
        parent.robot,
        agent.robot,
        device,
        loss_type=getattr(
            trans_args, "attention_distill_loss", "attention_kl_feature_mse"
        ),
        lambda_attention=getattr(
            trans_args, "attention_distill_lambda_a", 1.0
        ),
        lambda_feature=getattr(
            trans_args, "attention_distill_lambda_h", 1.0
        ),
    )
    return child_controller


def _has_active_stage(obs):
    return "stage" not in obs or bool((obs["stage"] > 0).any().item())


def _detach_obs(obs):
    return {
        key: value.detach().cpu().clone()
        for key, value in obs.items()
        if key != "design"
    }


def _stack_obs(obs_list, device):
    result = {}
    for key in obs_list[0].keys():
        result[key] = torch.cat([obs[key] for obs in obs_list], dim=0).to(device)
    return result


def _token_order(robot):
    flat = np.asarray(robot).reshape(-1)
    return np.concatenate((np.flatnonzero(flat != 0), np.flatnonzero(flat == 0)))


def _build_distill_views(obs, parent_robot, child_robot, device):
    parent = {key: value.clone() for key, value in obs.items()}
    child = {key: value.clone() for key, value in obs.items()}
    batch_size = obs["modular"].shape[0]
    seq_size = np.asarray(parent_robot).size
    shared_flat = np.flatnonzero(same_voxel_mask(parent_robot, child_robot))

    parent_order = obs["token_index"][0].long().cpu().numpy()
    child_order = _token_order(child_robot)
    parent_lookup = {int(flat): idx for idx, flat in enumerate(parent_order)}
    child_lookup = {int(flat): idx for idx, flat in enumerate(child_order)}
    parent_indices = torch.tensor(
        [parent_lookup[int(flat)] for flat in shared_flat], dtype=torch.long, device=device
    )
    child_indices = torch.tensor(
        [child_lookup[int(flat)] for flat in shared_flat], dtype=torch.long, device=device
    )

    parent_padding = torch.ones(seq_size, dtype=torch.bool, device=device)
    child_padding = torch.ones(seq_size, dtype=torch.bool, device=device)
    parent_padding[parent_indices] = False
    child_padding[child_indices] = False
    parent["obs_mask"] = parent_padding.unsqueeze(0).repeat(batch_size, 1).float()
    child["obs_mask"] = child_padding.unsqueeze(0).repeat(batch_size, 1).float()

    for key in ("modular", "daab_modular"):
        if key not in obs:
            continue
        feature_size = obs[key].shape[1] // seq_size
        source = obs[key].reshape(batch_size, seq_size, feature_size)
        parent_tokens = torch.zeros_like(source)
        child_tokens = torch.zeros_like(source)
        parent_tokens[:, parent_indices] = source[:, parent_indices]
        child_tokens[:, child_indices] = source[:, parent_indices]
        parent[key] = parent_tokens.reshape(batch_size, -1)
        child[key] = child_tokens.reshape(batch_size, -1)

    child_flat = np.asarray(child_robot).reshape(-1)
    child_types = torch.as_tensor(
        child_flat[child_order], dtype=obs["voxel_type"].dtype, device=device
    )
    child_rows, child_cols = np.unravel_index(child_order, np.asarray(child_robot).shape)
    child_grid = torch.as_tensor(
        np.stack((child_cols, child_rows), axis=1).reshape(-1),
        dtype=obs["grid_pos"].dtype, device=device,
    )
    child_token_index = torch.as_tensor(
        child_order, dtype=obs["token_index"].dtype, device=device
    )
    child["voxel_type"] = child_types.unsqueeze(0).repeat(batch_size, 1)
    child["grid_pos"] = child_grid.unsqueeze(0).repeat(batch_size, 1)
    child["token_index"] = child_token_index.unsqueeze(0).repeat(batch_size, 1)
    return parent, child, parent_indices, child_indices


def _robot_padding(robot, device):
    return torch.tensor(np.asarray(robot).reshape(-1) == 0, dtype=torch.bool, device=device)


def _same_attention_mask(same_mask):
    seq_size = same_mask.shape[0]
    mask = torch.zeros((seq_size, seq_size), dtype=torch.bool, device=same_mask.device)
    mask[same_mask] = ~same_mask.unsqueeze(0).expand(int(same_mask.sum().item()), -1)
    return mask


def _column_importance(attn_weights, indices, eps=1e-8):
    attn = attn_weights[-1]
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    same = attn.index_select(1, indices).index_select(2, indices)
    scores = same.mean(dim=1)
    return scores / (scores.sum(dim=-1, keepdim=True) + eps)


def _kl_divergence(parent_scores, child_scores, eps=1e-8):
    parent_scores = parent_scores.clamp_min(eps)
    child_scores = child_scores.clamp_min(eps)
    return (parent_scores * (parent_scores.log() - child_scores.log())).sum(dim=-1).mean()


def _shared_branch_loss(parent_attn, child_attn, parent_hidden, child_hidden,
                        parent_indices, child_indices, loss_type,
                        lambda_attention, lambda_feature):
    loss = torch.zeros((), dtype=torch.float32)
    if loss_type in SHARED_ATTENTION_LOSSES:
        loss = loss + lambda_attention * _shared_attention_kl(
            parent_attn, child_attn, parent_indices, child_indices
        )
    if loss_type in SHARED_FEATURE_LOSSES:
        loss = loss + lambda_feature * _shared_feature_loss(
            parent_hidden, child_hidden, parent_indices, child_indices
        )
    return loss


def _validate_distillation_config(loss_type, lambda_attention, lambda_feature):
    valid_loss_types = set(DISTILL_LOSS_TYPES)
    if loss_type not in valid_loss_types:
        raise ValueError(
            "Unknown attention distillation loss {!r}; expected one of {}".format(
                loss_type, sorted(valid_loss_types)
            )
        )
    if lambda_attention < 0 or lambda_feature < 0:
        raise ValueError("Attention distillation loss weights must be non-negative")


def _attention_with_head_dimension(attention):
    if attention.dim() == 3:
        return attention.unsqueeze(1)
    if attention.dim() != 4:
        raise ValueError(
            "Expected attention shaped [B, H, Q, K] or [B, Q, K], got {}".format(
                tuple(attention.shape)
            )
        )
    return attention


def _shared_attention_kl(parent_attention, child_attention,
                         parent_indices, child_indices, eps=1e-8):
    if len(parent_attention) != len(child_attention) or not parent_attention:
        raise ValueError("Parent and child must expose the same non-zero number of layers")

    layer_losses = []
    for parent_layer, child_layer in zip(parent_attention, child_attention):
        parent_layer = _attention_with_head_dimension(parent_layer)
        child_layer = _attention_with_head_dimension(child_layer)
        parent_shared = parent_layer.index_select(-2, parent_indices).index_select(
            -1, parent_indices
        ).detach()
        child_shared = child_layer.index_select(-2, child_indices).index_select(
            -1, child_indices
        )
        if parent_shared.shape != child_shared.shape:
            raise ValueError(
                "Parent and child shared attention shapes differ: {} vs {}".format(
                    tuple(parent_shared.shape), tuple(child_shared.shape)
                )
            )
        parent_shared = parent_shared / (
            parent_shared.sum(dim=-1, keepdim=True) + eps
        )
        child_shared = child_shared / (
            child_shared.sum(dim=-1, keepdim=True) + eps
        )
        parent_shared = parent_shared.clamp_min(eps)
        child_shared = child_shared.clamp_min(eps)
        layer_losses.append(
            (
                parent_shared
                * (parent_shared.log() - child_shared.log())
            ).sum(dim=-1).mean()
        )
    return torch.stack(layer_losses).mean()


def _shared_feature_loss(parent_hidden, child_hidden,
                         parent_indices, child_indices):
    if len(parent_hidden) != len(child_hidden) or not parent_hidden:
        raise ValueError("Parent and child must expose the same non-zero number of layers")

    layer_losses = []
    for parent_layer, child_layer in zip(parent_hidden, child_hidden):
        parent_shared = parent_layer.index_select(0, parent_indices).permute(1, 0, 2)
        child_shared = child_layer.index_select(0, child_indices).permute(1, 0, 2)
        if parent_shared.shape != child_shared.shape:
            raise ValueError(
                "Parent and child shared feature shapes differ: {} vs {}".format(
                    tuple(parent_shared.shape), tuple(child_shared.shape)
                )
            )
        normalized_parent = F.layer_norm(
            parent_shared.detach(), (parent_shared.shape[-1],)
        )
        normalized_child = F.layer_norm(
            child_shared, (child_shared.shape[-1],)
        )
        layer_losses.append(
            (normalized_parent - normalized_child).pow(2).sum(dim=-1).mean()
        )
    return torch.stack(layer_losses).mean()


def _enable_qk_layernorm(controller):
    trainable = []
    hooks = []
    for name, param in controller.named_parameters():
        if "self_attn.in_proj" in name:
            param.requires_grad_(True)
            trainable.append(param)
            embed_dim = param.shape[0] // 3
            mask = torch.zeros_like(param)
            mask[: 2 * embed_dim] = 1
            hooks.append(param.register_hook(lambda grad, mask=mask: grad * mask))
        elif "self_attn.q_proj." in name or "self_attn.k_proj." in name or "norm" in name:
            param.requires_grad_(True)
            trainable.append(param)
    return trainable, hooks
