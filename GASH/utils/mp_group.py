import multiprocessing
import time
import traceback
from utils.algo_utils import is_promote, write_start_log

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

    def __init__(self):
        
        self.jobs = []
        self.return_data = []
        self.callback = []
        self.expr_name=None
        self.gen=None
        self.params=None
        
    
    def add_args(self,expr_name,gen,params):
        self.expr_name=expr_name
        self.gen=gen
        self.params=params


    def add_job(self, func, args, callback):

        self.return_data.append(multiprocessing.Value("d", 0.0))
        self.jobs.append(multiprocessing.Process(target=job_wrapper, args=(func, args, self.return_data[-1])))
        self.callback.append(callback)

    def run_jobs(self, num_proc,queue):
        
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
            

            #while((num_jobs_open < num_proc or (queue.qsize() < num_proc and is_promote(self.expr_name,self.gen,self.params,0))) and next_job != len(self.jobs)):
            while((num_jobs_open < num_proc or (queue.qsize() < num_proc)) and next_job != len(self.jobs)):
                #write_start_log(self.expr_name,self.gen,0,self.params)
                self.jobs[next_job].start()
                jobs_open.add(next_job)
                queue.put(0) 
                next_job += 1
                num_jobs_open += 1
                print('queue length = {}, num_jobs_open = {}'.format(queue.qsize(),num_jobs_open))
                #time.sleep(1)

            time.sleep(0.1)

        for i in range(len(self.jobs)):
            self.callback[i](self.return_data[i].value)



