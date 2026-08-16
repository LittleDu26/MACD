from evogym import is_connected, has_actuator
from utils.algo_utils import mutate
from evogym import hashable,sample_robot
import random
import numpy as np
import copy
from .Agent import Agent
def eval_robot_constraint(robot):
    validity = is_connected(robot) and has_actuator(robot)
    return validity

def fast_non_dominated_sort(agents):
    fronts = [[]]
    domination_count = {}
    dominated_solutions = {}

    for p in agents:
        dominated_solutions[p] = []
        domination_count[p] = 0
        for q in agents:
            if p == q:
                continue
            if dominates(p, q):
                dominated_solutions[p].append(q)
            elif dominates(q, p):
                domination_count[p] += 1

        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts):
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        if next_front:
            fronts.append(next_front)
        i += 1

    for front in fronts:
        sort_front_by_rank_sum(front)

    return fronts


def sort_front_by_rank_sum(front):
    """
    Sort one non-dominated front by the sum of two within-front ranks:
    higher fitness ranks earlier, lower maturity ranks earlier.
    """
    fitness_order = sorted(front, key=lambda a: a.fitness, reverse=True)
    maturity_order = sorted(front, key=lambda a: a.maturity)
    fitness_rank = {agent: rank for rank, agent in enumerate(fitness_order, start=1)}
    maturity_rank = {agent: rank for rank, agent in enumerate(maturity_order, start=1)}
    front.sort(
        key=lambda a: (
            fitness_rank[a] + maturity_rank[a],
            fitness_rank[a],
            maturity_rank[a],
        )
    )


def dominates(a, b):
    better_or_equal = (a.fitness >= b.fitness) and (a.maturity <= b.maturity)
    strictly_better = (a.fitness > b.fitness) or (a.maturity < b.maturity)
    return better_or_equal and strictly_better

def keep_highest_n_agents(fronts, n):
    total_kept = 0
    new_fronts = []

    for front in fronts:
        if total_kept + len(front) <= n:
            new_fronts.append(front[:]) 
            total_kept += len(front)
        else:
            
            num_to_keep = n - total_kept
            new_fronts.append(front[:num_to_keep])
            break  

    return new_fronts


def print_fronts(fronts):
    for i, front in enumerate(fronts):
        print(f"Front {i}:")
        for agent in front:
            print(f"  Fitness: {agent.fitness:.2f}, Maturity: {agent.maturity}")

def get_fitness_median(new_fronts):
    fitness_list = [agent.fitness for front in new_fronts for agent in front if agent.fitness is not None]
    median = np.median(fitness_list)
    return median

def get_similar_index(ind, off):
    ind_structure = np.array(ind.robot)
    off_structures = np.array([agent.robot for agent in off])

    dif = off_structures - ind_structure 
    zero_num = np.zeros(dif.shape[0])     
    for i in range(len(zero_num)):
        zero_num[i] = np.sum(dif[i] == 0)
    index = np.argmax(zero_num)  
    return index

# 获得与child_agent结构最相似的parent_agent，相似度通过相同非零位置的数量衡量
def get_attention_distill_parent(child_agent, parent_list):
    best_parent = None
    best_count = -1
    child_robot = np.array(child_agent.robot).reshape(-1)
    for parent in parent_list:
        if getattr(parent, "best_controller", None) is None:
            continue
        parent_robot = np.array(parent.robot).reshape(-1)
        same_count = int(np.sum((parent_robot == child_robot) & (parent_robot != 0)))
        if same_count > best_count:
            best_parent = parent
            best_count = same_count
    return best_parent


def mutate_top(fronts, record, total_mutations=10):
    new_fronts = copy.deepcopy(fronts)
    pool = [agent for front in new_fronts for agent in front]
    top_layer = new_fronts[0]
    rest_agents = [agent for front in new_fronts[1:] for agent in front]
    random.shuffle(rest_agents)
    agents_to_mutate = top_layer + rest_agents
    mutated_children = []

    for agent in agents_to_mutate:
        child_robot = mutate(agent.robot, record)
        if child_robot is not None:
            new_agent = Agent(child_robot)
            partner_index = get_similar_index(new_agent, pool)
            agent.controller = pool[partner_index].controller
            mutated_children.append(new_agent)
        if len(mutated_children) >= total_mutations:
            break
    return mutated_children


# ============================================================
# Archive-Prior Grammar-Guided Mutation Operator
# ============================================================

def _get_4neighbors(x, y, H, W):
    """Return valid 4-connected neighbors of (x, y) within grid bounds."""
    result = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < H and 0 <= ny < W:
            result.append((nx, ny))
    return result


def compute_archive_prior(elite_set, H, W, num_types=5, eps=1e-3, tau=1.0):
    """
    Compute archive prior P[H][W][num_types] from a set of elite agents.

    P[x][y][t] = (eps + sum_i w_i * I[s_i[x,y]==t]) / (num_types*eps + sum_i w_i)

    This guarantees sum_t P[x][y][t] == 1 for every position.
    """
    if not elite_set:
        return np.full((H, W, num_types), 1.0 / num_types)

    fitnesses = np.array([agent.fitness for agent in elite_set], dtype=float)
    mean_f = fitnesses.mean()
    std_f = fitnesses.std()
    f_norm = (fitnesses - mean_f) / (std_f + 1e-8)
    weights = np.exp(f_norm / tau)
    total_w = float(weights.sum())

    P = np.full((H, W, num_types), eps)
    for agent, w in zip(elite_set, weights):
        for x in range(H):
            for y in range(W):
                t = int(agent.robot[x, y])
                if 0 <= t < num_types:
                    P[x, y, t] += w

    P /= (num_types * eps + total_w)
    return P


def enumerate_candidates(robot, rule, max_actuator_ratio=0.6):
    """
    Enumerate all feasible (c, t) candidate operations for a grammar rule.

    Returns a list of ((x, y), t) pairs.
    """
    H, W = robot.shape
    candidates = []

    if rule == "boundary_growth":
        # Empty cells adjacent to the body; target types {1,2,3,4}
        for x in range(H):
            for y in range(W):
                if robot[x, y] == 0:
                    if any(robot[nx, ny] != 0
                           for nx, ny in _get_4neighbors(x, y, H, W)):
                        for t in (1, 2, 3, 4):
                            candidates.append(((x, y), t))

    elif rule == "actuator_specialization":
        # Passive voxels → actuator, subject to max actuator ratio
        total_cells = int(np.sum(robot != 0))
        actuator_count = int(np.sum((robot == 3) | (robot == 4)))
        for x in range(H):
            for y in range(W):
                if robot[x, y] in (1, 2):
                    if total_cells == 0 or (actuator_count + 1) / total_cells <= max_actuator_ratio:
                        candidates.append(((x, y), 3))
                        candidates.append(((x, y), 4))

    elif rule == "tip_pruning":
        # Passive voxels with ≤1 non-empty neighbor; must keep connectivity + actuator
        for x in range(H):
            for y in range(W):
                if robot[x, y] in (1, 2):
                    nonempty_nb = [
                        (nx, ny) for nx, ny in _get_4neighbors(x, y, H, W)
                        if robot[nx, ny] != 0
                    ]
                    if len(nonempty_nb) <= 1:
                        test = robot.copy()
                        test[x, y] = 0
                        if is_connected(test) and has_actuator(test):
                            candidates.append(((x, y), 0))

    elif rule == "support_thickening":
        # Empty cells adjacent to weak-support voxels; target types {1,2}
        weak = set()
        for x in range(H):
            for y in range(W):
                if robot[x, y] != 0:
                    nb_count = sum(
                        1 for nx, ny in _get_4neighbors(x, y, H, W)
                        if robot[nx, ny] != 0
                    )
                    if nb_count <= 2:
                        weak.add((x, y))
        for x in range(H):
            for y in range(W):
                if robot[x, y] == 0:
                    neighbors = _get_4neighbors(x, y, H, W)
                    if any(robot[nx, ny] != 0 for nx, ny in neighbors):
                        if any((nx, ny) in weak for nx, ny in neighbors):
                            candidates.append(((x, y), 1))
                            candidates.append(((x, y), 2))

    elif rule == "actuator_refinement":
        # Refine existing actuators by de-specializing, flipping direction,
        # or pruning low-connectivity actuator tips when globally feasible.
        for x in range(H):
            for y in range(W):
                if robot[x, y] in (3, 4):
                    candidates.append(((x, y), 1))
                    candidates.append(((x, y), 2))
                    candidates.append(((x, y), 4 if robot[x, y] == 3 else 3))

                    nonempty_nb = [
                        (nx, ny) for nx, ny in _get_4neighbors(x, y, H, W)
                        if robot[nx, ny] != 0
                    ]
                    if len(nonempty_nb) <= 1:
                        test = robot.copy()
                        test[x, y] = 0
                        if is_connected(test) and has_actuator(test):
                            candidates.append(((x, y), 0))

    return candidates


def sample_operation_by_archive_prior(robot, rule, P):
    """
    Sample a (c, t) operation from the candidate set weighted by P[c][t].

    Returns ((x, y), t) or None if no candidates exist.
    """
    candidates = enumerate_candidates(robot, rule)
    if not candidates:
        return None

    scores = np.array(
        [max(float(P[c[0], c[1], t]), 1e-12) for c, t in candidates]
    )
    probs = scores / scores.sum()
    idx = int(np.random.choice(len(candidates), p=probs))
    return candidates[idx]


def sample_operations_by_archive_prior(robot, rule, P, k=3):
    """
    Sample up to k distinct (c, t) operations weighted by P[c][t].

    Returns a list of ((x, y), t) pairs. Sampling is without replacement so a
    high-prior duplicate operation cannot consume every slot in one trial.
    """
    candidates = enumerate_candidates(robot, rule)
    if not candidates:
        return []

    sample_size = min(k, len(candidates))
    scores = np.array(
        [max(float(P[c[0], c[1], t]), 1e-12) for c, t in candidates]
    )
    probs = scores / scores.sum()
    indices = np.random.choice(len(candidates), size=sample_size, replace=False, p=probs)
    return [candidates[int(idx)] for idx in indices]


def apply_grammar_mutation(robot, c, t):
    """Return a deep copy of robot with cell c set to type t."""
    child = copy.deepcopy(robot)
    child[c[0], c[1]] = t
    return child


def grammar_feasibility_check(robot, record, min_voxels=3, max_voxels=None,
                               max_actuator_ratio=0.6):
    """
    Return True iff the robot passes all structural feasibility constraints
    and has not been seen before.
    """
    H, W = robot.shape
    if max_voxels is None:
        max_voxels = H * W
    nonempty = int(np.sum(robot != 0))
    if nonempty < min_voxels or nonempty > max_voxels:
        return False
    if not is_connected(robot) or not has_actuator(robot):
        return False
    actuator_count = int(np.sum((robot == 3) | (robot == 4)))
    if actuator_count / nonempty > max_actuator_ratio:
        return False
    if hashable(robot) in record:
        return False
    return True


def grammar_guided_mutate(Parents, record, total_offspring, H, W, P,
                           inherit=False, max_trials=25, rule_probs=None,
                           selection_alpha=0.5,
                           operations_per_rule=3):
    """
    Generate offspring via Archive-Prior Grammar-Guided Mutation.

    Each offspring is produced by:
      1. Taking the next parent from the already shuffled Parents list.
      2. Trying every grammar rule once on that parent.
      3. Sampling up to operations_per_rule operations per rule weighted by P[c][t].
      4. Keeping feasible candidates.
      5. Selecting one candidate by rule_weight * P[c][t]^selection_alpha.
      6. Falling back to random mutation if max_trials exceeded.

    Returns (children_list, child_logs).
    """
    if not Parents or total_offspring <= 0:
        return [], []

    if rule_probs is None:
        rule_probs = {
            "boundary_growth": 0.25,
            "actuator_specialization": 0.20,
            "support_thickening": 0.15,
            "tip_pruning": 0.15,
            "actuator_refinement": 0.25,
        }

    rules = list(rule_probs.keys())
    rule_weights = np.array([rule_probs[r] for r in rules], dtype=float)
    rule_weights /= rule_weights.sum()

    children = []
    child_logs = []
    parent_attempts = 0
    max_parent_attempts = min(len(Parents), total_offspring)

    for parent in Parents:
        if len(children) >= total_offspring or parent_attempts >= max_parent_attempts:
            break
        parent_attempts += 1

        child_robot = None
        selected_rule = None
        selected_c = None
        selected_t = None
        prior_val = None

        for _ in range(max_trials):
            candidates = []
            scores = []

            for rule, rule_weight in zip(rules, rule_weights):
                ops = sample_operations_by_archive_prior(
                    parent.robot, rule, P, k=operations_per_rule)

                for c, t in ops:
                    candidate = apply_grammar_mutation(parent.robot, c, t)
                    if grammar_feasibility_check(candidate, record):
                        p_val = max(float(P[c[0], c[1], t]), 1e-12)
                        candidates.append((candidate, rule, c, t, p_val))
                        scores.append(rule_weight * (p_val ** selection_alpha))

            if candidates:
                probs = np.array(scores, dtype=float)
                probs /= probs.sum()
                selected_idx = int(np.random.choice(len(candidates), p=probs))
                child_robot, selected_rule, selected_c, selected_t, prior_val = candidates[selected_idx]
                break

        if child_robot is None:
            fallback_robot = mutate(parent.robot, record)
            if fallback_robot is not None:
                child_robot = fallback_robot
                selected_rule = "random_fallback"
                selected_c = None
                selected_t = None
                prior_val = None

        if child_robot is not None:
            if hashable(child_robot) not in record:
                record[hashable(child_robot)] = []
            new_agent = Agent(child_robot)
            if inherit:
                new_agent.distill_parent = get_attention_distill_parent(new_agent, Parents)
                # new_agent.maturity = max(parent.maturity//2, 0)
            children.append(new_agent)
            child_logs.append({
                "child_id": new_agent.id,
                "parent_id": parent.id,
                "rule": selected_rule,
                "position": selected_c,
                "target_type": selected_t,
                "prior_value": prior_val,
                "feasible": True,
            })

    return children, child_logs


def _pop_random_agent(candidates):
    if not candidates:
        return None
    return candidates.pop(random.randrange(len(candidates)))


def random_mutate_offspring(survivors, final_candidates, record, total_offspring,
                            inherit=False, final_ratio=0.30,
                            structure_shape=(5,5)):
    """
    Generate offspring with a fixed split:
    final_ratio of parents from final_candidates, the rest from survivors.
    Each source pool is sampled without replacement.
    """
    if total_offspring <= 0:
        return [], []

    survivor_pool = list(survivors)
    final_pool = list(final_candidates)
    if not survivor_pool and not final_pool:
        return [], []

    final_needed = int(total_offspring * final_ratio)
    # If final pool is smaller than needed, reduce final count first.
    final_needed = min(final_needed, len(final_pool))
    survivor_needed = total_offspring - final_needed
    # Then cap survivors by available pool size.
    survivor_needed = min(survivor_needed, len(survivor_pool))

    random.shuffle(final_pool)
    random.shuffle(survivor_pool)
    parent_list = final_pool[:final_needed] + survivor_pool[:survivor_needed]+final_pool[final_needed:] + survivor_pool[survivor_needed:]

    children = []
    child_logs = []
    for parent in parent_list:
        if len(children)==total_offspring:
            break
        child_robot = mutate(parent.robot, record)
        if child_robot is None:
            continue

        new_agent = Agent(child_robot)
        if inherit:
            new_agent.distill_parent = get_attention_distill_parent(new_agent, parent_list)
            # new_agent.maturity = max(parent.maturity//2, 0)
        children.append(new_agent)
    
    while len(children)<total_offspring:
        robot, _ = sample_robot(structure_shape)
        while (hashable(robot) in record or not eval_robot_constraint(robot)):
            robot, _ = sample_robot(structure_shape)
        new_agent = Agent(robot)
        if inherit:
            new_agent.distill_parent = get_attention_distill_parent(new_agent, parent_list)
        children.append(new_agent)
        record[hashable(robot)] = []
    return children, child_logs


def get_best_agent(population):
    return max(population, key=lambda agent: agent.fitness)

def check_children(population):
    for agent in population:
        if not is_connected(agent.robot):
            print("ERROR")
            return agent
    return None
def check_size(population,target_size):
    for agent in population:
        if agent.robot.shape[0]>target_size or agent.robot.shape[1]>target_size:
            return agent
    return None

def get_par(env_name):
    tc = 0
    max_eva = 0
    if env_name == 'Walker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BridgeWalker-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'BidirectionalWalker-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'Carrier-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Carrier-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Pusher-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Pusher-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Thrower-v0':
        tc = 300
        max_eva = 150
    elif env_name == 'Catcher-v0':
        tc = 400
        max_eva = 200
    elif env_name == 'BeamToppler-v0':
        tc = 1000
        max_eva = 100
    elif env_name == 'BeamSlider-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Lifter-v0':
        tc = 300
        max_eva = 200
    elif env_name == 'Climber-v0':
        tc = 400
        max_eva = 150
    elif env_name == 'Climber-v1':
        tc = 600
        max_eva = 150
    elif env_name == 'Climber-v2':
        tc = 1000
        max_eva = 200
    elif env_name == 'UpStepper-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'DownStepper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'ObstacleTraverser-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'ObstacleTraverser-v1':
        tc = 1000
        max_eva = 200
    elif env_name == 'Hurdler-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'PlatformJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'GapJumper-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'Traverser-v0':
        tc = 1000
        max_eva = 200
    elif env_name == 'CaveCrawler-v0':
        tc = 1000
        max_eva = 150
    elif env_name == 'AreaMaximizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'AreaMinimizer-v0':
        tc = 600
        max_eva = 150
    elif env_name == 'WingspanMazimizer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'HeightMaximizer-v0':
        tc = 500
        max_eva = 150
    elif env_name == 'Flipper-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Jumper-v0':
        tc = 500
        max_eva = 100
    elif env_name == 'Balancer-v0':
        tc = 600
        max_eva = 100
    elif env_name == 'Balancer-v1':
        tc = 600
        max_eva = 150

    return max_eva, tc
