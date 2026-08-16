import datetime

class CustomReporter():
    def __init__(self, save_to):
        self.generation = None
        self.generation_start_time = None
        self.generation_times = []
        self.num_extinctions = 0
        self.txt = open(save_to +"/out.txt", "w")

    def start_generation(self, generation):
        self.generation = generation
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        text = ""
        text += '\n ****** [{1}] Running generation {0} ****** \n'.format(generation, now)
        print(text)
        self.txt.write(text)
        self.txt.flush()

    def end_generation(self, population, all_children, child_logs, Survivors):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pop_size = len(population)
        population.sort(key=lambda agent: agent.fitness, reverse=True)
        best_agent = population[0]
        text = ""
        text += f"[{now}]Population of {pop_size} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ===   =======\n"
        for agent in population:
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)
        text += "Start maturity-stage promotion selection\n"
        text += "Survivors:" + str(len(Survivors)) + "\n"
        text += f"survivor IDs({len(Survivors)}):" + str([s.id for s in Survivors]) + "\n"
        text += f"children IDs({len(all_children)}):" + str([c.id for c in all_children]) + "\n"
        # text += f"Child generation log ({len(child_logs)} children):\n"
        # text += "  {:>8s} {:>9s} {:>24s} {:>10s} {:>8s} {:>10s}\n".format(
        #     "child_id", "parent_id", "rule", "pos", "type_t", "P[c][t]")
        # for log in child_logs:
        #     pos_str = str(log["position"]) if log["position"] is not None else "N/A"
        #     t_str = str(log["target_type"]) if log["target_type"] is not None else "N/A"
        #     p_str = f"{log['prior_value']:.4f}" if log["prior_value"] is not None else "N/A"
        #     text += "  {:>8d} {:>9d} {:>24s} {:>10s} {:>8s} {:>10s}\n".format(
        #         log["child_id"], log["parent_id"], log["rule"], pos_str, t_str, p_str)
        self.txt.write(text)
        self.txt.flush()
        print(text)

    def start_elites(self,population):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        pop_size = len(population)
        best_agent=population[0]
        text = ""
        text += f"[{now}]Population of {pop_size} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ========   =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)

        text += '\n ****** [{0}] Elites training ****** \n'.format(now)
        print(text)
        self.txt.write(text)
        self.txt.flush()

    def end_elites(self,population):
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # 当前时间戳
        best_agent = population[0]
        text = ""
        text += f"[{now}]Population of {len(population)} members\tbest_id:(ID{best_agent.id},{best_agent.fitness})\n"
        text += "   ID   maturity   fitness\n"
        text += "  ====  ========   =======\n"
        for i, agent in enumerate(population):
            text += "  {0:>4d}  {1:>3d}   {2:>7.4f}\n".format(agent.id, agent.maturity, agent.fitness)
        self.txt.write(text)
        self.txt.flush()
        print(text)
