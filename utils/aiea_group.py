import torch.multiprocessing as mp
import time
import traceback

def job_wrapper(func, args, data_container):
        try:
            out_value = func(*args)
        except:
            print("ERROR\n")
            traceback.print_exc()
            print()
            return 0   
        data_container.value = out_value

class Group():

    def __init__(self, reward=None, num_evaluations=None):
        
        self.jobs = []
        self.return_data = []
        self.callback = []
        if reward == None:
            self.reward = {}
        else:
            self.reward = reward
        if num_evaluations == None:
            self.num_evaluations = 0
        else:
            self.num_evaluations = num_evaluations


    def add_job(self, func, args, callback=None):
        ctx = mp.get_context("spawn")
        self.return_data.append(ctx.Value("d", 0.0))
        self.jobs.append(ctx.Process(target=job_wrapper, args=(func, args, self.return_data[-1])))
        if callback != None:
            self.callback.append(callback)

    def run_jobs(self, num_proc):
        
        next_job = 0
        num_jobs_open = 0
        jobs_finished = 0

        jobs_open = set()

        while(jobs_finished != len(self.jobs)):

            jobs_closed = []
            for job_index in jobs_open:
                if not self.jobs[job_index].is_alive():
                    self.jobs[job_index].join()
                    self.jobs[job_index].terminate()
                    num_jobs_open -= 1
                    jobs_finished += 1
                    jobs_closed.append(job_index)

            for job_index in jobs_closed:
                jobs_open.remove(job_index)

            while(num_jobs_open < num_proc and next_job != len(self.jobs)):
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                next_job += 1
                num_jobs_open += 1

            time.sleep(0.1)

        for i in range(len(self.jobs)):
            if len(self.callback) != 0:
                self.callback[i](self.return_data[i].value)
            else:
                self.reward[str(i+self.num_evaluations)] = self.return_data[i].value



