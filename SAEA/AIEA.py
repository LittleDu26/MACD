import copy
import os
import numpy as np
import pandas as pd
import sys
import csv
import gym
import torch
import math
from ppo.run import run_ppo
from ppo.envs import make_vec_envs
from evogym import sample_robot, hashable
import utils.aiea_group as mp
from utils.algo_utils import ga_mutate, TerminationCondition, get_percent_survival_evals
from ppo.utils import get_vec_normalize


curr_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(curr_dir, '..')
external_dir = os.path.join(root_dir, 'externals')
sys.path.insert(0, root_dir)
sys.path.insert(1, os.path.join(external_dir, 'pytorch_a2c_ppo_acktr_gail'))


def variation(parent, par_fit, record, pop_size):
    par_size = len(parent)
    off_structure = []
    off_connection = []
    i = 0
    while i < pop_size:
        # 二元锦标赛选择
        r = np.random.choice(np.arange(0, par_size), 2, replace=False)
        r1 = r[0]
        r2 = r[1]
        p1 = parent[r1]
        p2 = parent[r2]
        if par_fit[r1] > par_fit[r2]:
            p = copy.deepcopy(p1)
        else:
            p = copy.deepcopy(p2)
        tmp_child = ga_mutate(p, mutation_rate = 0.1, num_attempts=50)
        if tmp_child != None and (hashable(tmp_child[0]) not in record):
            off_structure.append(tmp_child[0])
            off_connection.append(tmp_child[1])
            i = i + 1
    return off_structure, off_connection


def local_search(ind, record, norb_size):
    off_structure = []
    off_connection = []
    tmp_hashes = {}
    i = 0
    while i < norb_size:
        tmp_child = ga_mutate(copy.deepcopy(ind), mutation_rate = 0.1, num_attempts=50)
        while tmp_child != None and (hashable(tmp_child[0]) not in record) and (hashable(tmp_child[0]) not in tmp_hashes):
            off_structure.append(tmp_child[0])
            off_connection.append(tmp_child[1])
            tmp_hashes[hashable(tmp_child[0])] = True
            i = i + 1
            break
    return off_structure, off_connection


def evaluate_by_action(ind, action, env_name):
    act_len = action.shape[0]
    env = gym.make(env_name, body=ind)
    env.reset()
    sum_reward = 0
    i = 0
    # 给指令添加一部分，因为在实验中发现，指令和控制器之间存在一些误差
    anchor = int(4*act_len/5)
    action = np.vstack((action, action[anchor:act_len,:]))
    while i < act_len:
        ob, reward, done, info = env.step(action[i])
        sum_reward = sum_reward + reward
        if done == True:
            break
        i = i + 1
    env.close()
    return sum_reward


def evaluate_by_transfered_action(off, sim_par_action, env_name, mask, dif_num):
    action_len = len(mask)
    action_group = len(sim_par_action)

    off_action = np.zeros((action_group, action_len))
    for j in range(len(mask)):
        m = mask[j]
        if m == -1:
            off_action[:, j] = np.random.uniform(-1.6, 1.6, size=(1, action_group))
        else:
            off_action[:, j] = sim_par_action[:, int(m)]
    reward = evaluate_by_action(off, off_action, env_name)
    return reward


## 获得模型的输出，训练模型用
## reward的类型是字典，把它转换成数组
def get_output(reward):
    output = np.zeros(len(reward))
    for i in reward.keys():
        output[int(i)] = reward[i]
    return output


## 根据real_fits从大到小，只拿出前pop_size个个体
def maintain_pop(pop_genome, real_fits, max_id, pop_size):
    parent = []
    parent_fits = []
    ## 当前排序结果是从小到大
    sorted_id = np.argsort(real_fits)
    i = len(sorted_id) - 1
    while len(parent) < pop_size:
        sid = sorted_id[i]
        if sid != max_id:
            parent.append(pop_genome[sid])
            parent_fits.append(real_fits[sid])
        i = i - 1
    return parent, parent_fits

def get_similar_id(ind, off):
    dif = np.array(off) - np.array(ind)
    zero_num = np.zeros(dif.shape[0])
    for i in range(len(zero_num)):
        zero_num[i] = len(np.where(dif[i] == 0)[0])
    id = np.argmax(zero_num)
    return id


def save_actions(action_path, controller_path, structure, connection, env_name, label):

    agent = tuple([structure, connection])
    actor_critic, obs_rms = torch.load(controller_path + '/robot_' + str(label) + '_controller.pt', map_location='cpu')

    env = make_vec_envs(env_name, agent, 1, 1, None, None, device='cpu', allow_early_resets=False)

    vec_norm = get_vec_normalize(env)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.obs_rms = obs_rms

    recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
    masks = torch.zeros(1, 1)
    obs = env.reset()

    actions = []
    done = False

    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)
        obs, reward, done, _ = env.step(action)
        ## 保存数据
        actions.append(np.array(action)[0])
        masks.fill_(0.0 if (done) else 1.0)
    env.venv.close()
    tmp_path = action_path+'/robot_'+str(label)+'_action.csv'
    with open(tmp_path, 'a+', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        writer.writerows(actions)


def integration(par_structure, off_structure):
    par_genemo = [i for itm in par_structure for i in itm]
    off_genemo = [i for itm in off_structure for i in itm]
    par_genemo = np.array(par_genemo)
    off_genemo = np.array(off_genemo)
    par_actor_indx = np.where(par_genemo >= 3)[0]
    off_actor_indx = np.where(off_genemo >= 3)[0]
    par_action_len = len(par_actor_indx)
    off_action_len = len(off_actor_indx)
    mask = np.zeros(off_action_len) - 1
    i = 0
    j = 0
    while i < off_action_len and j < par_action_len:
        oai = off_actor_indx[i]
        pai = par_actor_indx[j]
        if oai < pai:
            i = i + 1
        elif oai > pai:
            j = j + 1
        elif oai == pai and off_genemo[oai] == par_genemo[pai]:
            mask[i] = j
            i = i + 1
            j = j + 1
        elif oai == pai and off_genemo[oai] != par_genemo[pai]:
            i = i + 1
            j = j + 1
    dif_num = len(np.where(mask == -1)[0])
    return mask, dif_num


def error_evaluate(par_structure, par_action, par_fit, env_name):
    ite_num = par_action.shape[0]
    action_len = par_action.shape[1]
    errors = []
    for i in range(action_len):
        # lb = np.min(par_action[:, i])
        # ub = np.max(par_action[:, i])
        tmp_action = copy.deepcopy(par_action)
        tmp_action[:, i] = np.random.uniform(-1.6, 1.6, size=(1, ite_num))
        tmp_reward = evaluate_by_action(par_structure, tmp_action, env_name)
        if tmp_reward > par_fit / 3:
            errors.append(par_fit - tmp_reward)
    if len(errors) == 0:
        return 0
    else:
        error = np.mean(errors)
        return error

def get_off_fits(off, pop, pop_fits, action_path, env_name, record, gen_off_path=None):
    original_fits = []
    fixed_fits = []
    dif_num = []
    # out = ""
    i = 0
    sel_results = []
    for o in off:
        ## 1、拿到相似个体的ID
        similar_id = get_similar_id(o, pop)
        par_structure = pop[similar_id]
        par_fit = pop_fits[similar_id]
        ## 2、根据ID读取指令
        action_data = pd.read_csv(action_path+'/robot_'+str(similar_id)+'_action.csv', sep=',', header=None)
        par_action = action_data.values
        ## 3、使用父代的指令控制子代，得到奖励
        mask, dif = integration(par_structure, o)
        ori_fit = evaluate_by_transfered_action(o, par_action, env_name, mask, dif)
        hash_key = hashable(par_structure)
        saved_error = record[hash_key]
        if len(saved_error) == 0:
            error = error_evaluate(par_structure, par_action, par_fit, env_name)
            record[hash_key] = [error]
        else:
            error = saved_error[0]
        increment = error
        fixed_fit = ori_fit + increment
        original_fits.append(ori_fit)
        fixed_fits.append(fixed_fit)
        dif_num.append(dif)
        if ori_fit >= par_fit or fixed_fit > par_fit:
            sel_results.append(1)
        else:
            sel_results.append(0)
        i = i + 1
    return original_fits, fixed_fits, dif_num, sel_results

def get_local_surviving(parent_fit, local_fits, local_dif):
    tmp_fits = np.array(local_fits)
    tmp_sur = np.where(tmp_fits >= parent_fit)[0]
    if len(tmp_sur) <= 1:
        tmp_dif = np.array(local_dif)
        zero_pos = np.where(tmp_dif == 0)[0]
        not_zero_pos = np.where(tmp_dif != 0)[0]
        zero_fits = tmp_fits[zero_pos]
        not_zero_fits = tmp_fits[not_zero_pos]
        max_id1 = zero_pos[np.argmax(zero_fits)]
        max_id2 = not_zero_pos[np.argmax(not_zero_fits)]
        env_sel = np.array([max_id1, max_id2])
    else:
        env_sel = tmp_sur
    return env_sel

def get_variation_surviving(par_fits, off_fits, num_survivors):
    combined_fits = list(par_fits) + list(off_fits)
    tmp_fits = 100 - np.array(combined_fits)
    survivals = []
    pop_size = len(par_fits)
    sorted_id = np.argsort(tmp_fits)
    for i in range(pop_size):
        si = sorted_id[i]
        if si >= pop_size:
            survivals.append(si-pop_size)
    if len(survivals) < num_survivors:
        return survivals
    else:
        return survivals[0:num_survivors]


def sel_best(fits, dif, k):
    tmp_fits = 100 - np.array(fits)
    zero_indx = np.where(np.array(dif) == 0)[0]
    if len(zero_indx) < k:
        sorted_indx = np.argsort(tmp_fits)
        sel_result = []
        i = 0
        while len(sel_result) < k:
            sel_result.append(sorted_indx[i])
            i = i + 1
    else:
        sel_result = []
        for j in range(2):
            if j == 0:
                part_indx = np.where(np.array(dif) == 0)[0]
                tmp_k = int(k / 2)
            else:
                part_indx = np.where(np.array(dif) != 0)[0]
                tmp_k = k - int(k / 2)
            part_fits = tmp_fits[part_indx]
            sorted_indx = np.argsort(part_fits)
            i = 0
            while len(sel_result) < tmp_k:
                sel_result.append(sorted_indx[i])
                i = i + 1
    return np.array(sel_result)

def sel_random(fits, k):
    pop_size = len(fits)
    index = np.arange(0, pop_size)
    sel_result = np.random.choice(index, k, replace=False)
    return sel_result



## Action Sampling based Evolutionary Algorithm
def run_aiea(pop_size, structure_shape, max_evaluations, train_iters, num_cores, env_name, save_path,seed,max_reward=None):
    home_path = save_path
    ## 强化学习的终止条件
    tc = TerminationCondition(train_iters)
    ## 迭代次数
    gen = 0
    ## 优化中所有个体的哈希表
    record = {}
    ## 记录最优解没有发生变化的迭代次数
    count = 0

    ## 随机初始化初代种群
    # 初始化种群
    pop_structure = []
    pop_connection = []
    for i in range(pop_size):
        structure, connection = sample_robot(structure_shape)
        while (hashable(structure) in record):
            structure, connection = sample_robot(structure_shape)
        pop_structure.append(structure)
        pop_connection.append(connection)
        record[hashable(structure)] = []

    ## 先创建保存数据的文件夹
    eva_path = os.path.join(home_path, 'real_eva')
    controler_path = os.path.join(eva_path, 'controller')
    structure_path = os.path.join(eva_path, 'structure')
    action_path = os.path.join(eva_path, 'action')

    try:
        os.makedirs(controler_path)
        os.makedirs(structure_path)
        os.makedirs(action_path)
    except:
        pass

    # 评价初代种群
    group = mp.Group()
    for i in range(len(pop_structure)):
        ppo_args = ((pop_structure[i], pop_connection[i]), tc, (controler_path, i), env_name, seed)
        group.add_job(run_ppo, ppo_args)
        temp_path = os.path.join(structure_path, str(i))
        np.savez(temp_path, pop_structure[i], pop_connection[i])
    group.run_jobs(num_cores)
    rewards = group.reward
    pop_fit = get_output(rewards)

    ## 需要单独保存actions
    for i in range(len(pop_structure)):
        save_actions(action_path, controler_path, pop_structure[i], pop_connection[i], env_name, i)

    ## 更新当前使用的评价次数
    num_evaluations = len(pop_fit)

    ## 获得训练模型的输入和输出数据
    max_id = np.argmax(pop_fit)
    local_search_id = [max_id]

    ## 初始化父代种群和父代适应度值
    parent = copy.deepcopy(pop_structure)
    parent_fits = copy.deepcopy(pop_fit)

    ## 写入所有真实评估的个体的适应度值，随时保存结果
    optima_path = os.path.join(home_path, 'generations.csv')
    with open(optima_path, 'w', encoding='utf-8') as fp:
        writer = csv.writer(fp)
        for pfi in range(len(pop_fit)):
            writer.writerow([pfi, pop_fit[pfi]])

    gen = gen + 1

    while num_evaluations < max_evaluations:

        percent_survival = get_percent_survival_evals(num_evaluations, max_evaluations)
        num_survivors = max(4, math.ceil(pop_size * percent_survival))

        ## 先创建保存子代的文件夹
        gen_path = os.path.join(home_path, 'generation_' + str(gen))
        gen_structure_path = os.path.join(gen_path, 'structure')

        try:
            os.makedirs(gen_structure_path)
        except:
            pass

        # 根据交叉变异产生子代
        variation_structure, variation_connection = variation(parent, parent_fits, record, pop_size)
        variation_size = len(variation_structure)
        for i in range(variation_size):
            temp_path = os.path.join(gen_structure_path, str(i))
            np.savez(temp_path, variation_structure[i], variation_connection[i])

        # 根据局部搜索产生子代
        local_structure, local_connection = local_search(pop_structure[max_id], record,pop_size)
        local_size = len(local_structure)
        for i in range(local_size):
            temp_path = os.path.join(gen_structure_path, str(variation_size+i))
            np.savez(temp_path, local_structure[i], local_connection[i])

        ## 基于指令继承的近似评估
        variation_original_fits, variation_fixed_fits, variation_dif, variation_sel = get_off_fits(variation_structure, pop_structure, pop_fit, action_path, env_name, record, gen_structure_path+"var.txt")
        local_original_fits, local_fixed_fits, local_dif, local_sel = get_off_fits(local_structure, pop_structure, pop_fit, action_path, env_name, record, gen_structure_path+"local.txt")

        combine_structure = variation_structure + local_structure
        combine_connection = variation_connection + local_connection
        combine_fits = variation_fixed_fits + local_fixed_fits
        combine_sel = variation_sel + local_sel

        env_sel_results = np.where(np.array(combine_sel) == 1)[0]
        if len(env_sel_results) > num_survivors:
            tmp_fits = 100 - np.array(combine_fits)[env_sel_results]
            sel_indx = np.argsort(tmp_fits)[0:num_survivors]
            env_sel_results = env_sel_results[sel_indx]
        if len(env_sel_results) < 4:
            env_sel_result1 = sel_best(variation_fixed_fits, variation_dif, num_survivors)
            env_sel_result2 = sel_best(local_fixed_fits, local_dif, num_survivors)
            env_sel_results = list(env_sel_results) + list(env_sel_result1) + list(env_sel_result2 + variation_size)
            env_sel_results = np.unique(np.array(env_sel_results))

        # 评价选出的采样个体
        group = mp.Group()
        for index, value in enumerate(env_sel_results):
            reeva_stru = combine_structure[value]
            reeva_conn = combine_connection[value]
            ppo_args = ((reeva_stru, reeva_conn), tc, (controler_path, num_evaluations + index), env_name, seed)
            group.add_job(run_ppo, ppo_args)
            ## 保存数据，包括写入外部的数据和写入缓存的数据
            record[hashable(reeva_stru)] = []
            ## 写入外部数据，方便生成gif图片
            temp_path = os.path.join(structure_path, str(num_evaluations + index))
            np.savez(temp_path, reeva_stru, reeva_conn)
            ## 写入缓存的数据
            pop_structure.append(reeva_stru)
        group.run_jobs(num_cores)
        real_fits = get_output(group.reward)

        for index, value in enumerate(env_sel_results):
            save_actions(action_path, controler_path, combine_structure[value], combine_connection[value], env_name, num_evaluations+index)

        if np.max(real_fits) < np.max(pop_fit):
            count = count + 1
        else:
            count = 0

        pop_fit = np.append(pop_fit, real_fits)
        num_evaluations = len(pop_fit)

        sorted_indx = np.argsort(100-pop_fit)
        ti = 0
        while sorted_indx[ti] in local_search_id:
            ti = ti + 1
        max_id = sorted_indx[ti]
        local_search_id.append(max_id)

        ## 维持种群
        parent, parent_fits = maintain_pop(pop_structure, pop_fit, max_id, pop_size)
        gen = gen + 1

        ## 写入所有真实评估的个体的适应度值，方便画图
        optima_path = os.path.join(home_path, 'generations.csv')
        with open(optima_path, 'w', encoding='utf-8') as fp:
            writer = csv.writer(fp)
            for rfi in range(len(pop_fit)):
                writer.writerow([rfi, pop_fit[rfi]])


        if max_reward != None and np.max(pop_fit) >= max_reward:
            break