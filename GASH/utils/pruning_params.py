import math

class Singleton():
    def __new__(cls, *args, **kargs):
        if not hasattr(cls,"_instance"):
            cls._instance = super().__new__(cls)
        return cls._instance

class Params(Singleton):

    def __init__(self,pop_size,eval_timing_arr,scale=1,num_cores=32) -> None:
        self.pruning_num=None
        self.params=None
        self.eval_require_dict=None
        self.eval_border_dict=None
        self.num_cores=num_cores
        self.eval_timing_arr=eval_timing_arr
        self.calc_params_interactivly(pop_size,scale)

    def calc_params_interactivly(self,eval_pop_size,scale=1):
        n=math.ceil(math.log2(eval_pop_size//scale))
        self.pruning_num=n-1

        self.params={}
        self.eval_require_dict={}
        self.eval_border_dict={}
        self.params['timing0_border']=eval_pop_size

        pre_index=5-n
        for i in range(self.pruning_num):
            self.params["pruning_timing"+str(i+1)]=self.eval_timing_arr[pre_index+i]
            self.params["timing"+str(i+1)+"_border"]=math.ceil(self.params["timing"+str(i)+"_border"]/2.0)
            self.eval_require_dict[self.params["pruning_timing"+str(i+1)]]=self.params["timing"+str(i)+"_border"]
            self.eval_border_dict[self.params["pruning_timing"+str(i+1)]]=self.params["timing"+str(i+1)+"_border"]
    

    def judge_timing(self,j):
        for i in range(1,self.pruning_num+1):
            index="pruning_timing"+str(i)
            if j==self.params[index]:
                return True
        return False

if __name__=="__main__":
    #params=Params()
    #print('id: {} \n'.format(id(params)))
    #params=Params()
    #print('id: {} \n'.format(id(params)))
    '''
    for i in range(12,33):
        params.calc_params_interactivly(i)
        print(params.params)
        for j in range(50,1000,50):
            if params.judge_timing(j):
                print(j)
    '''

'''
pruning_num=4
params = \
    {
        "timing0_border": 32,
        "pruning_timing1": 50,
        "pruning_timing2": 100,
        'pruning_timing3': 250,
        'pruning_timing4': 500,
        "timing1_border": 16,
        "timing2_border": 8,
        "timing3_border": 4,
        "timing4_border": 2,
    }

eval_require_dict={}
eval_border_dict={}

eval_require_dict[params["pruning_timing1"]]=params["timing0_border"]
eval_require_dict[params["pruning_timing2"]]=params["timing1_border"]
eval_require_dict[params["pruning_timing3"]]=params["timing2_border"]
eval_require_dict[params["pruning_timing4"]]=params["timing3_border"]

eval_border_dict[params["pruning_timing1"]]=params["timing1_border"]
eval_border_dict[params["pruning_timing2"]]=params["timing2_border"]
eval_border_dict[params["pruning_timing3"]]=params["timing3_border"]
eval_border_dict[params["pruning_timing4"]]=params["timing4_border"]
pruning_num=2

params = \
    {
        "timing0_border": 4,
        "pruning_timing1": 50,
        "pruning_timing2": 100,
        "timing1_border": 2,
        "timing2_border": 1,
    }

eval_require_dict={}
eval_border_dict={}

eval_require_dict[params["pruning_timing1"]]=params["timing0_border"]
eval_require_dict[params["pruning_timing2"]]=params["timing1_border"]


eval_border_dict[params["pruning_timing1"]]=params["timing1_border"]
eval_border_dict[params["pruning_timing2"]]=params["timing2_border"]
'''




