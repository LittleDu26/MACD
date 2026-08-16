import copy
import math
import numpy as np
from evogym import is_connected, has_actuator, get_full_connectivity, draw, get_uniform,hashable

class Structure():

    def __init__(self, body, connections, label):
        self.body = body
        self.connections = connections

        self.reward = 0
        self.fitness = self.compute_fitness()
        
        self.is_survivor = False
        self.prev_gen_label = 0

        self.label = label

    def compute_fitness(self):

        self.fitness = self.reward
        return self.fitness

    def set_reward(self, reward):

        self.reward = reward
        self.compute_fitness()

    def __str__(self):
        return f'\n\nStructure:\n{self.body}\nF: {self.fitness}\tR: {self.reward}\tID: {self.label}'

    def __repr__(self):
        return self.__str__()

class TerminationCondition():

    def __init__(self, max_iters):
        self.max_iters = max_iters

    def __call__(self, iters):
        return iters >= self.max_iters

    def change_target(self, max_iters):
        self.max_iters = max_iters


def ga_mutate(child, mutation_rate=0.1, num_attempts=10):

    pd = get_uniform(5)
    pd[0] = 0.6 #it is 3X more likely for a cell to become empty

    # iterate until valid robot found
    for n in range(num_attempts):
        # for every cell there is mutation_rate% chance of mutation
        for i in range(child.shape[0]):
            for j in range(child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    child[i][j] = draw(pd)

        if is_connected(child) and has_actuator(child):
            return (child, get_full_connectivity(child))

    # no valid robot found after num_attempts
    return None



def mutate(child, record,mutation_rate=0.1, num_attempts=25):
    new_child=copy.deepcopy(child)
    pd = get_uniform(5)
    pd[0] = 0.4 #it is 3X more likely for a cell to become empty

    # iterate until valid robot found
    for n in range(num_attempts):
        # for every cell there is mutation_rate% chance of mutation
        for i in range(new_child.shape[0]):
            for j in range(new_child.shape[1]):
                mutation = [mutation_rate, 1-mutation_rate]
                if draw(mutation) == 0: # mutation
                    new_child[i][j] = draw(pd)
        
        if is_connected(new_child) and has_actuator(new_child) and (hashable(new_child) not in record):
            record[hashable(new_child)] = []
            return new_child

    # no valid robot found after num_attempts
    # print("ERROR:mutate")
    return None



## num_attempts，最大迭代次数
def ls_mutate(child, mutation_rate=0.1, frquency = None):
    pd = get_uniform(5)
    pd[0] = 0.6  # it is 3X more likely for a cell to become empty

    # for every cell there is mutation_rate% chance of mutation
    for i in range(child.shape[0]):
        for j in range(child.shape[1]):
            if np.random.random() <= mutation_rate:  # mutation
                if frquency == None:
                    mutated_voxel = draw(pd)
                else:
                    pd = [i*5+j]
                    mutated_voxel = draw(pd)
                ## 如果变异的voxel为0，需要判断robot是否是联通的
                if mutated_voxel == 0:
                    child[i][j] = mutated_voxel
                    if is_connected(child) == False:
                        child[i][j] = np.random.randint(1, 5)
                else:
                    child[i][j] = mutated_voxel
    ## 判断robot内是否有actor
    actor_position = np.where(np.array(child) >= 3)
    if len(actor_position[0]) == 0:
        p1 = np.where(np.array(child) == 1)
        p2 = np.where(np.array(child) == 2)
        x_ax = list(p1[0]) + list(p2[0])
        y_ax = list(p1[1]) + list(p2[1])
        indx = np.random.randint(0, len(x_ax))
        child[x_ax[indx]][y_ax[indx]] = np.random.randint(3, 5)
    return child, get_full_connectivity(child)

def get_percent_survival(gen, max_gen):
    low = 0.0
    high = 0.8
    return ((max_gen-gen-1)/(max_gen-1))**1.5 * (high-low) + low

def total_robots_explored(pop_size, max_gen):
    total = pop_size
    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
    return total

def total_robots_explored_breakpoints(pop_size, max_gen, max_evaluations):
    
    total = pop_size
    out = []
    out.append(total)

    for i in range(1, max_gen):
        total += pop_size - max(2, math.ceil(pop_size * get_percent_survival(i-1, max_gen))) 
        if total > max_evaluations:
            total = max_evaluations
        out.append(total)

    return out

def search_max_gen_target(pop_size, evaluations):
    target = 0
    while total_robots_explored(pop_size, target) < evaluations:
        target += 1
    return target
    


def parse_range(str_inp, rbt_max):
    
    inp_with_spaces = ""
    out = []
    
    for token in str_inp:
        if token == "-":
            inp_with_spaces += " " + token + " "
        else:
            inp_with_spaces += token
    
    tokens = inp_with_spaces.split()

    count = 0
    while count < len(tokens):
        if (count+1) < len(tokens) and tokens[count].isnumeric() and tokens[count+1] == "-":
            curr = tokens[count]
            last = rbt_max
            if (count+2) < len(tokens) and tokens[count+2].isnumeric():
                last = tokens[count+2]
            for i in range (int(curr), int(last)+1):
                out.append(i)
            count += 3
        else:
            if tokens[count].isnumeric():
                out.append(int(tokens[count]))
            count += 1
    return out

def pretty_print(list_org, max_name_length=30):

    list_formatted = []
    for i in range(len(list_org)//4 +1):
        list_formatted.append([])

    for i in range(len(list_org)):
        row = i%(len(list_org)//4 +1)
        list_formatted[row].append(list_org[i])

    print()
    for row in list_formatted:
        out = ""
        for el in row:
            out += str(el) + " "*(max_name_length - len(str(el)))
        print(out)

def get_percent_survival_evals(curr_eval, max_evals):
    low = 0.1
    high = 0.6
    return ((max_evals-curr_eval-1)/(max_evals-1)) * (high-low) + low

def total_robots_explored_breakpoints_evals(pop_size, max_evals):
    
    num_evals = pop_size
    out = []
    out.append(num_evals)
    while num_evals < max_evals:
        num_survivors = max(2,  math.ceil(pop_size*get_percent_survival_evals(num_evals, max_evals)))
        new_robots = pop_size - num_survivors
        num_evals += new_robots
        if num_evals > max_evals:
            num_evals = max_evals
        out.append(num_evals)

    return out


def structure_to_genome(structure):
    return [i for itm in structure for i in itm]


def crossover(p1, p2, rate=0.2, num_attempts=20):
    off = []
    t1 = structure_to_genome(p1)
    t2 = structure_to_genome(p2)
    dim = len(t1)
    cros_len = int(dim*rate)
    for n in range(num_attempts):
        ran_pos = np.random.randint(0, dim-cros_len)
        for j in range(cros_len):
            tmp = t1[ran_pos + j]
            t1[ran_pos + j] = t2[ran_pos + j]
            t2[ran_pos + j] = tmp
        o1 = np.array(t1).reshape((5, 5))
        o2 = np.array(t2).reshape((5, 5))
        if is_connected(o1) and has_actuator(o1) and len(off) < 2:
            off.append(o1)
        if is_connected(o2) and has_actuator(o2) and len(off) < 2:
            off.append(o2)
        if len(off) >= 2:
            break
    return off

def heuristic_crossover(p1, p2,record,random_rate=0.5,num_attempts=10):
    target_size=p1.shape[0]
    g1 = structure_to_genome(p1)
    g2 = structure_to_genome(p2)

    for i in range(num_attempts):
        child_genome = []
        for gene1, gene2 in zip(g1, g2):
            if gene1 == gene2:
                child_genome.append(gene1)
            else:
                if np.random.rand() < random_rate:
                    child_genome.append(gene1)
                else:
                    child_genome.append(gene2)
        child = np.array(child_genome).reshape((target_size, target_size))

        if is_connected(child) and has_actuator(child) and (hashable(child) not in record):
            record[hashable(child)] = []
            return child
    else:
        print("ERROR:heuristic_crossover")
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