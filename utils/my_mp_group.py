import torch
import torch.multiprocessing as mp
import os
import shutil
import time
import traceback
import tempfile
from typing import Any, List


def job_wrapper(func, args, result_path):
    """
    子进程执行的包装函数，将结果对象写入独立文件。

    不通过 Manager().list() 返回 Torch 模型/张量对象。Manager 代理在反序列化
    Torch storage 时会走共享内存 mmap，长时间并行训练后容易触发 Cannot allocate
    memory。
    """
    try:
        out_value = func(*args)
    except Exception:
        print("ERROR\n")
        traceback.print_exc()
        print()
        out_value = None

    try:
        torch.save(out_value, result_path)
    except Exception:
        print("ERROR saving multiprocessing result\n")
        traceback.print_exc()
        print()


class Group():

    def __init__(self):
        self.jobs = []
        self.return_paths = []
        self.result_dir = tempfile.mkdtemp(prefix="agcd_mp_results_")
        self.callback = []

    def add_job(self, func, args, callback=None):
        ctx = mp.get_context("spawn")
        result_path = os.path.join(self.result_dir, "{}.pt".format(len(self.jobs)))
        self.return_paths.append(result_path)
        self.jobs.append(ctx.Process(target=job_wrapper, args=(func, args, result_path)))
        if callback is not None:
            self.callback.append(callback)

    def run_jobs(self, num_proc) -> List[Any]:
        next_job = 0
        num_jobs_open = 0
        jobs_finished = 0
        jobs_open = set()

        while jobs_finished != len(self.jobs):

            # 检查已完成的子进程
            jobs_closed = []
            for job_index in jobs_open:
                if not self.jobs[job_index].is_alive():
                    self.jobs[job_index].join()
                    num_jobs_open -= 1
                    jobs_finished += 1
                    jobs_closed.append(job_index)

            for job_index in jobs_closed:
                jobs_open.remove(job_index)

            # 启动新子进程
            while num_jobs_open < num_proc and next_job != len(self.jobs):
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                next_job += 1
                num_jobs_open += 1

            # 等待所有任务完成
            time.sleep(0.1)

        results = []
        try:
            for job_index, result_path in enumerate(self.return_paths):
                if os.path.exists(result_path):
                    results.append(torch.load(result_path, map_location="cpu"))
                else:
                    print("ERROR: multiprocessing job {} produced no result".format(job_index))
                    results.append(None)
            return results
        finally:
            shutil.rmtree(self.result_dir, ignore_errors=True)
