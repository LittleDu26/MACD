import csv
import datetime
class CustomReporter():
    def __init__(self, save_to):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.txt = open(save_to +"/out.txt", "w")
        self.csv = open(save_to+"/table.csv", "w")
        self.csv_logger = csv.DictWriter(self.csv, fieldnames=('generation','pop_size','best_fit'))
        self.csv_logger.writeheader()

    def start_generation(self, generation):
        self.generation = generation
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        text = ""
        text += '\n ****** [{1}] Running generation {0} ****** \n'.format(generation, now) + "\n"
        print(text)
        self.txt.write(text)
        self.txt.flush()

    def end_generation(self, population):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        pop_size = len(population)
        csv_content = {"generation": self.generation,"pop_size":pop_size, "best_fit":population[0].fitness}
        self.csv_logger.writerow(csv_content)
        self.csv.flush()
        text = ""
        text += f"[{now}]Population of {pop_size} members,best_fit:{population[0].fitness}\n"
        text += "   ID   fitness\n"
        text += "  ====  =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>7.4f}\n".format(i, agent.fitness)
        self.txt.write(text)
        self.txt.flush()
        print(text)

