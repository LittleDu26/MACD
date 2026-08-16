class Agent():
    _id_counter = 1  
    def __init__(self,robot):
        self.id = Agent._id_counter  
        self.robot=robot
        self.maturity=0
        self.birth=0
        self.initial_fitness=0
        self.fitness=float('-inf')
        self.history_fitness=[]
        self.iteration=0
        self.controller=None
        self.best_controller=None
        self.finished=False
        self.structure_saved=False
        self.distill_parent=None
        Agent._id_counter += 1
