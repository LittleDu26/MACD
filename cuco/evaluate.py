import multiprocessing as mp
import os
import queue
import signal
import numpy as np
import torch
from . import helper
from .envs import make_vec_envs
from utils.algo_utils import Structure
from evogym import get_full_connectivity

EVALUATION_TIMEOUT_SECONDS = 180


def _evaluate_once(
    num_evals, 
    uni_agent, 
    ob_rms,
    env_name,
    init_robots, 
    init_nca_design=None,
    seed=1, 
    nca_setting=None,
    device=None):

    eval_envs = make_vec_envs(
        env_name,
        init_robots,
        seed,
        None,
        device,
        ret=False,
        ob=True,
        nca_setting=nca_setting,
        init_nca_design=init_nca_design,
    )
    
    vec_norm = helper.get_vec_normalize(eval_envs)
    
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    # recorders
    final_robot = None
    eval_episode_rewards = []
    obs = eval_envs.reset()

    while True:
        with torch.no_grad():
            val, action, logp = uni_agent.uni_act(obs, mean_action=True)
        obs, rewards, done, infos = eval_envs.step(action)
        for info in infos:
            if 'episode' in info.keys():
                eval_episode_rewards.append(info['episode']['r'])

            if info["design_success"]=='Done':
                final_design = info['real_design']
                final_robot = Structure(body=final_design,connections=get_full_connectivity(final_design), label=777)

        if num_evals*len(init_robots) == len(eval_episode_rewards):
            break
    eval_envs.close()
    print("Evalution done!")
    return np.average(eval_episode_rewards), final_robot


def _evaluation_worker(result_queue, args, kwargs):
    # Make the evaluator and its EvoGym workers a killable process group.
    try:
        os.setsid()
    except OSError:
        pass

    try:
        result_queue.put(("ok", _evaluate_once(*args, **kwargs)))
    except BaseException as error:
        result_queue.put(("error", repr(error)))


def evaluate(*args, **kwargs):
    """Run evaluation in an isolated process so a stuck simulator cannot hang PPO."""
    context = mp.get_context("fork")
    result_queue = context.Queue(maxsize=1)
    evaluator = context.Process(target=_evaluation_worker, args=(result_queue, args, kwargs))
    evaluator.start()
    evaluator.join(EVALUATION_TIMEOUT_SECONDS)

    if evaluator.is_alive():
        print(f"Evaluation timed out after {EVALUATION_TIMEOUT_SECONDS}s; terminating evaluator.")
        try:
            os.killpg(evaluator.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        evaluator.join(10)
        if evaluator.is_alive():
            try:
                os.killpg(evaluator.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            evaluator.join()
        result_queue.close()
        return 0.0, None

    try:
        status, payload = result_queue.get(timeout=1)
    except queue.Empty:
        print("Evaluation exited without a result.")
        result_queue.close()
        return 0.0, None

    result_queue.close()
    if status == "ok":
        return payload

    print(f"Evaluation failed: {payload}")
    return 0.0, None

   
