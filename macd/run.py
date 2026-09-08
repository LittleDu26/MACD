import os
import csv
import torch
from utils.MyUtils import *
from evogym import sample_robot, hashable, get_full_connectivity
from utils.Agent import Agent
from .Reporter import CustomReporter
from .transformer.config import transformerconfig, ppoconfig
import gym
from .transformer.transformerPPOagent import PPOAgent, TransformerPPOAC
from .ppo import PPO
from .controller_distillation import prepare_distilled_controller
import copy
import torch.multiprocessing as mlp
import evogym.envs
import utils.my_mp_group as mp
import json
import numpy as np


def _max_maturity_stage(total_step, train_iters):
    """Return the number of maturity stages needed to consume ``total_step``."""
    if total_step <= 0:
        raise ValueError("total_step must be positive")
    if train_iters <= 0:
        raise ValueError("train_iters must be positive")
    return (int(total_step) + int(train_iters) - 1) // int(train_iters)


def _allocated_stage_iters(agent_iteration, total_step, train_iters,
                           current_iters, max_iters):
    """Compute a stage length without exceeding either training budget."""
    if train_iters <= 0:
        raise ValueError("train_iters must be positive")
    agent_remaining = max(0, int(total_step) - int(agent_iteration))
    global_remaining = max(0, int(max_iters) - int(current_iters))
    return min(int(train_iters), agent_remaining, global_remaining)


def single_agent_fit(agent, ppo_args, trans_args, sample_setting, args,
                     stage_iters):
    # Seed
    device = torch.device("cpu")
    torch.set_num_threads(1)

    robot = (agent.robot, get_full_connectivity(agent.robot))

    if agent.controller is not None:
        ppoAgent = agent.controller
    else:
        ppoAgent = None
        if args.distill and agent.distill_parent:
            ppoAgent = prepare_distilled_controller(agent, ppo_args, trans_args, sample_setting, args, device)
        if ppoAgent is None:
            # Init PPO Actor-Critic
            actor_critic = TransformerPPOAC(modular_state_dim=sample_setting[0], modular_action_dim=sample_setting[1],
                                            sequence_size=sample_setting[3], other_feature_size=sample_setting[2],
                                            ppo_args=ppo_args, trans_args=trans_args, ac_type="transformer",
                                            controller_type=trans_args.controller_type, device=device)
            # ppoagent
            ppoAgent = PPOAgent(actor_critic=actor_critic)
        agent.controller = ppoAgent

    ppoAgent.ac.to(device)

    # PPO
    # Each stage may be shorter than the configured stage length (the final
    # maturity or the final global-budget allocation).  Keep this override
    # local to the worker so parallel jobs cannot affect one another.
    ppo_args = copy.deepcopy(ppo_args)
    ppo_args.eval_interval = stage_iters
    ppo = PPO(robot=robot, ppo_size=1, train_iters=stage_iters, agent=ppoAgent, verbose=True,
              ppo_args=ppo_args, total_step=args.total_step, save_path=args.save_to, device=device,
              mmse=args.mmse)

    if agent.iteration==0:
        reward_history, agent.iteration, agent.initial_fitness= ppo.train(agent.id, agent.iteration, agent.fitness)
    else:
        reward_history, agent.iteration, _ = ppo.train(agent.id, agent.iteration,agent.fitness)

    if reward_history[-1] > agent.fitness:
        agent.fitness = reward_history[-1]
        agent.best_controller = copy.deepcopy(agent.controller)

    agent.history_fitness += reward_history
    agent.maturity += len(reward_history)
    if not args.mmse:
        agent.maturity = _max_maturity_stage(args.total_step, args.train_iters)
    # ppoAgent.ac.to("cpu")
    return agent


def make_dir(dir):
    os.makedirs(os.path.join(dir, "structures"))
    os.makedirs(os.path.join(dir, "controllers"))


def muti_running(agents, ppo_args, trans_args, sample_setting, args):
    global current_iters
    # 评价初代种群
    group = mp.Group()
    maturity_agents = []
    for agent in agents:
        if agent.finished:
            maturity_agents.append(agent)
        else:
            stage_iters = _allocated_stage_iters(
                agent.iteration,
                args.total_step,
                args.train_iters,
                current_iters,
                args.max_iters,
            )
            if stage_iters <= 0:
                break
            current_iters += stage_iters
            all_args = (agent, ppo_args, trans_args, sample_setting, args, stage_iters)
            group.add_job(single_agent_fit, all_args)
            robot = (agent.robot, get_full_connectivity(agent.robot))
            temp_path_body = os.path.join(args.save_to, "structures", str(agent.id))
            if not getattr(agent, "structure_saved", False):
                np.savez(temp_path_body, robot[0], robot[1])
                agent.structure_saved = True
    return maturity_agents + group.run_jobs(args.threads_num)


def _percentile_ranks(values):
    n = len(values)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        percentile = avg_rank / (n - 1)
        for k in range(i, j + 1):
            ranks[order[k]] = percentile
        i = j + 1
    return ranks


def _promotion_lambda(maturity, lambda_max, lambda_min, lambda_tau,
                      eps=1e-8, max_maturity_stage=25):
    tail = np.exp(-(max_maturity_stage - 1) / lambda_tau)
    decay = (
        np.exp(-(maturity - 1) / lambda_tau) - tail
    ) / (1.0 - tail + eps)
    return lambda_min + (lambda_max - lambda_min) * decay


def _last_training_delta(agent):
    if len(agent.history_fitness) >= 2:
        return agent.history_fitness[-1] - agent.history_fitness[-2]
    if len(agent.history_fitness) == 1:
        baseline = agent.initial_fitness
        if baseline is None:
            baseline = agent.history_fitness[-1]
        return agent.history_fitness[-1] - baseline
    return 0.0


def add_history_records(historical_archive, agent, generation, max_maturity_stage):
    history_len = len(agent.history_fitness)
    archived_len = getattr(agent, "archived_history_len", 0)
    if history_len <= archived_len:
        return []

    start_maturity = agent.maturity - history_len
    new_records = []
    for history_index in range(archived_len, history_len):
        maturity = start_maturity + history_index + 1
        fitness = agent.history_fitness[history_index]
        if history_index == 0:
            baseline = agent.initial_fitness
            if baseline is None:
                baseline = fitness
            previous_fitness = baseline
        else:
            previous_fitness = agent.history_fitness[history_index - 1]

        if maturity >= max_maturity_stage:
            status = "final"
        elif maturity < agent.maturity:
            status = "promoted"
        else:
            status = "active"

        record = {
            "agent_id": agent.id,
            "generation": generation,
            "maturity": maturity,
            "fitness": float(fitness),
            "delta": float(fitness - previous_fitness),
            "status": status,
        }
        historical_archive.setdefault(maturity, []).append(record)
        new_records.append(record)

    agent.archived_history_len = history_len
    return new_records


def add_history_record(historical_archive, agent, generation, max_maturity_stage):
    records = add_history_records(historical_archive, agent, generation, max_maturity_stage)
    return records[-1] if records else None


def mark_promoted_history(historical_archive, promoted_agents):
    promoted_ids = {agent.id for agent in promoted_agents}
    promoted_maturities = {agent.id: agent.maturity for agent in promoted_agents}
    for maturity, records in historical_archive.items():
        for record in reversed(records):
            agent_id = record["agent_id"]
            if (
                agent_id in promoted_ids
                and maturity == promoted_maturities[agent_id]
                and record["status"] == "active"
            ):
                record["status"] = "promoted"
                break


def _score_maturity_layer(records, lambda_max, lambda_min, lambda_tau, eps,
                          max_maturity_stage=25):
    fitness_values = [record["fitness"] for record in records]
    deltas = np.array([record["delta"] for record in records], dtype=float)
    percentile = _percentile_ranks(fitness_values)
    scale = float(np.median(np.abs(deltas)) + eps)
    maturity = records[0]["maturity"]
    lam = _promotion_lambda(
        maturity, lambda_max, lambda_min, lambda_tau, eps, max_maturity_stage)

    scores = {}
    values = []
    for record, f_rank, delta in zip(records, percentile, deltas):
        d_score = float(np.tanh(delta / scale))
        s_score = float(f_rank * (1.0 + lam * d_score))
        scores[id(record)] = s_score
        values.append(s_score)
    return scores, np.array(values, dtype=float)


def select_survivors(population, historical_archive, max_maturity_stage,
                     promotion_k=10, lambda_max=0.30, lambda_min=0.05,
                     lambda_tau=2.0, eps=1e-8):
    active_agents = [
        agent for agent in population
        if not agent.finished and agent.maturity < max_maturity_stage
    ]
    if not active_agents or promotion_k <= 0:
        return []

    active_by_layer = {}
    for agent in active_agents:
        active_by_layer.setdefault(agent.maturity, {})[agent.id] = agent

    candidates = []
    for maturity, records in historical_archive.items():
        layer_active = active_by_layer.get(maturity)
        if not layer_active or not records:
            continue

        score_by_record, layer_scores = _score_maturity_layer(
            records, lambda_max, lambda_min, lambda_tau, eps,
            max_maturity_stage)
        theta = float(np.quantile(layer_scores, 0.5))

        latest_active_records = {}
        for record in records:
            if record["status"] == "active" and record["agent_id"] in layer_active:
                latest_active_records[record["agent_id"]] = record

        for agent_id, record in latest_active_records.items():
            s_score = score_by_record[id(record)]
            if s_score >= theta:
                candidates.append((record["fitness"], s_score, layer_active[agent_id]))

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [agent for _, _, agent in candidates[:promotion_k]]


def run(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    global current_iters
    mlp.set_start_method('spawn', force=True)

    max_maturity_stage = _max_maturity_stage(args.total_step, args.train_iters)

    logger = CustomReporter(args.save_to)

    csv_file = open(args.save_to + "/table.csv", "w")
    csv_logger = csv.DictWriter(csv_file, fieldnames=('id', 'maturity', 'fit'))
    csv_logger.writeheader()

    trans_args = transformerconfig()
    ppo_args = ppoconfig()
    if trans_args.controller_type == "daab":
        trans_args.attention_heads = trans_args.daab_attention_heads
        trans_args.condition_decoder = True
    else:
        # CuCo's reordered modular observation contains 9 features per token,
        # so the old 4/4 axis split is no longer dimensionally valid.
        trans_args.use_separate_pos_embedding = False
    ppo_args.env_name = args.env
    ppo_args.seed = args.seed
    # The worker sets this again to its actual stage length.  Keeping a valid
    # default here also protects callers that inspect the PPO config directly.
    ppo_args.eval_interval = args.train_iters

    # 检查机器人
    structure_shape = (args.target_size, args.target_size)
    # 记录所有个体的哈希表
    record = {}
    pop_agent = []
    all_agents = {}
    historical_archive = {}
    # 重置Agent
    Agent._id_counter = 1
    generation = 0
    current_iters = 0
    fitness_window = []

    # Set dimensions
    robots = sample_robot(structure_shape)
    train_env = gym.make(args.env, mode='modular', body=robots[0], connections=robots[1], env_id=args.env)
    modular_state_dim = train_env.modular_state_dim
    modular_action_dim = train_env.modular_action_dim
    other_feature_size = train_env.other_dim
    sequence_size = train_env.voxel_num
    train_env.close()
    sample_setting = [modular_state_dim, modular_action_dim, other_feature_size, sequence_size]
    make_dir(args.save_to)
    # 初始化
    for _ in range(args.pop_size):
        robot, _ = sample_robot(structure_shape)
        while (hashable(robot) in record or not eval_robot_constraint(robot)):
            robot, _ = sample_robot(structure_shape)
        pop_agent.append(Agent(robot))
        record[hashable(robot)] = []

    while True:
        logger.start_generation(generation)
        pop_agent = muti_running(pop_agent, ppo_args, trans_args, sample_setting, args)

        for agent in pop_agent:
            if not agent.finished:
                if (args.mmse and agent.maturity == 1) or not args.mmse:
                    agent.birth = generation
                all_agents[agent.id] = agent
                if args.mmse:
                    fitness_window.append(agent)
                    if len(fitness_window) == max_maturity_stage:
                        best_agent = get_best_agent(fitness_window)
                        csv_content = {"id": best_agent.id, "maturity": best_agent.maturity, "fit": best_agent.fitness}
                        csv_logger.writerow(csv_content)
                        csv_file.flush()
                        fitness_window = []
                    add_history_records(historical_archive, agent, generation, max_maturity_stage)
                else:
                    csv_content = {"id": agent.id, "maturity": agent.maturity, "fit": agent.fitness}
                    csv_logger.writerow(csv_content)
                    csv_file.flush()
                if agent.iteration >= args.total_step:
                    agent.finished = True

        pop_agent.sort(key=lambda agent: agent.fitness, reverse=True)

        if current_iters >= args.max_iters:
            break

        if args.mmse:
            Survivors = select_survivors(
                pop_agent,
                historical_archive,
                max_maturity_stage,
                promotion_k=args.pop_size//2,
                lambda_max=getattr(args, "lambda_max", 0.30),
                lambda_min=getattr(args, "lambda_min", 0.05),
                lambda_tau=getattr(args, "lambda_tau", 2.0),
                eps=getattr(args, "selection_eps", 1e-8),
            )
            mark_promoted_history(historical_archive, Survivors)
        else:
            Survivors = pop_agent[:args.pop_size//2]
        
        child_num = args.pop_size - len(Survivors)
        if args.mmse:
            final_candidates = [
                agent for agent in all_agents.values()
                if agent.iteration >= args.total_step
            ]
            final_candidates.sort(key=lambda agent: agent.fitness, reverse=True)
            final_candidates = final_candidates[:max(1, len(final_candidates)//2)]
        else:
            final_candidates = []

        #mutation
        all_children, child_logs = random_mutate_offspring(
            Survivors, final_candidates, record, child_num, inherit=args.distill)

        logger.end_generation(pop_agent, all_children, child_logs, Survivors)
        pop_agent = Survivors + all_children

        generation += 1

    all_list = list(all_agents.values())
    all_list.sort(key=lambda a: a.fitness, reverse=True)
    finally_agent = all_list[:20]
    logger.end_elites(finally_agent)
    # 保存 all_list 为 json
    json_path = os.path.join(args.save_to, "all_agents.json")
    all_agents_json = [[agent.id, agent.birth, agent.maturity, agent.history_fitness] for agent in all_list]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_agents_json, f, ensure_ascii=False, indent=2)

    archive_path = os.path.join(args.save_to, "historical_archive.json")
    serializable_archive = {
        str(maturity): records
        for maturity, records in sorted(historical_archive.items())
    }
    with open(archive_path, "w", encoding="utf-8") as f:
        json.dump(serializable_archive, f, ensure_ascii=False, indent=2)
