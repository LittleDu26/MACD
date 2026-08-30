import torch
import torch.multiprocessing as mp
import os
import shutil
import time
import traceback
import tempfile
from typing import Any, List


def job_wrapper(input_path, result_path):
    """
    子进程执行的包装函数，从独立文件读取任务并将结果写入独立文件。

    使用 spawn 时，若直接把含 Tensor 的 ``args`` 传给 Process，PyTorch 会在
    ``Process.start()`` 期间为 storage 创建共享内存对象，容易耗尽文件描述符。
    因此 Process 的参数只传两个普通路径，任务对象在子进程启动后才从文件加载。
    """
    try:
        func, args = torch.load(input_path, map_location="cpu")
        out_value = func(*args)
    except Exception:
        print("ERROR running multiprocessing job\n")
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
        self.input_paths = []
        self.return_paths = []
        self.result_dir = tempfile.mkdtemp(prefix="agcd_mp_results_")
        self.callback = []

    def add_job(self, func, args, callback=None):
        ctx = mp.get_context("spawn")
        job_index = len(self.jobs)
        input_path = os.path.join(self.result_dir, "{}_input.pt".format(job_index))
        result_path = os.path.join(self.result_dir, "{}_result.pt".format(job_index))
        # Serialize tensors before starting the process.  Passing ``args`` directly
        # to ctx.Process would invoke torch.multiprocessing shared-memory reduction.
        torch.save((func, args), input_path)
        self.input_paths.append(input_path)
        self.return_paths.append(result_path)
        self.jobs.append(ctx.Process(target=job_wrapper, args=(input_path, result_path)))
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
