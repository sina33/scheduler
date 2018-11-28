from __future__ import print_function

import copy
from random import randint, random, randrange
from functools import reduce
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
logging.basicConfig(level=logging.INFO)

total_cores = 4
low_perf_multiplier = 2
task_graph = 'robot' # one of ['fpppp', 'sparse', 'robot']
stgs = ['fpppp', 'sparse', 'robot']

# Gets the index of maximum element in a list. If a conflict occurs, the index of the last largest is returned
def maxl(l): return l.index(reduce(lambda x,y: max(x,y), l))

# Gets the index of minimum element in a list. If a conflict occurs, the index of the last smallest is returned
def minl(l): return l.index(reduce(lambda x,y: min(x,y), l))

class Task:
    def __init__(self):
        self.deps = []
        self.id = 0
        self.exec_time = 0
        self.deadline = 0
        self.start_time = 0
        self.core = -1
        self.tg_id = 0

    def __repr__(self):
        if self.core == -1:
            return 'id: {} exec_time: {} deps: {}'.format(
                self.id,
                self.exec_time,
                [t.id for t in self.deps]
            )
        else:
           exec_time = self.exec_time * (low_perf_multiplier if self.core < total_cores / 2 else 1)
           return '{}: ({}, {})'.format(self.id, self.start_time, self.start_time+exec_time)

    def finish_time(self):
        exec_time = self.exec_time * (low_perf_multiplier if self.core < total_cores / 2 else 1)
        return self.start_time+exec_time


class Job:
    def __init__(self, task_id):
        self.start = 0
        self.id = task_id
        self.end = 0


class CoreQueue:
    def __init__(self, low=True):
        self.jobs = []
        self.last_job = 0
        self.low = low


def get_core_for_task(task, low_percent):
    if random() < low_percent:
        return randrange(0, total_cores / 2)
    return randrange(total_cores / 2, total_cores)


def create_individual(tasks, low_percent=0.75):
    """
    Create a member of the population.
    """
    schedule = [-1 for _ in range(len(tasks))]
    for task in tasks:
        schedule[task.id] = get_core_for_task(task, low_percent)

    return schedule


def create_population(tasks, population_size):
    """
    Create a number of individuals (i.e. a population).
    tasks: data structure holding tasks
    count: the number of individuals in the population
    """
    return [create_individual(tasks) for _ in range(population_size)]


def fitness_for_queue(core, queue):
    score = 0
    missed = 0
    sum_exec_time = 0
    for task in queue:
        exec_time = task.exec_time * (low_perf_multiplier if core < total_cores / 2 else 1)
        sum_exec_time += exec_time
        delta = task.deadline - (task.start_time + exec_time)
        if delta >= 0:  # deadline met
            score += delta 
        else:           # deadline missed
            score -= 10 * max(abs(delta), 100)
            missed += 1
            # print(task.id)
    # score += (sum_exec_time*2 if core < total_cores / 2 else 0)
    return score, missed


def get_makespan(tasks, schedule, print_out=False):
    new_tasks = copy.deepcopy(tasks)
    core_queues = [[] for _ in range(total_cores)]
    core_times = [0 for _ in range(total_cores)]
    for index in range(len(new_tasks)):
        # find the targeted core
        core = schedule[index]
        # append the task to the targeted core_queue
        core_queues[core].append(new_tasks[index])
        new_tasks[index].core = core
        # set the task's start_time to core's current time
        new_tasks[index].start_time = core_times[core] # + 1
        for dep in new_tasks[index].deps:
            # calculate end_time for task's dependencies
            dep_end_time = dep.start_time + dep.exec_time * (low_perf_multiplier if dep.core < total_cores / 2 else 1)
            # check if dep end_time is already passed
            if new_tasks[index].start_time < dep_end_time:
                new_tasks[index].start_time = dep_end_time
        exec_time = new_tasks[index].exec_time * (low_perf_multiplier if core < total_cores / 2 else 1)
        core_times[core] = new_tasks[index].start_time + exec_time

        # total execution time on each core        
        core_exec_sum = [0 for _ in range(total_cores)]
        for core, queue in enumerate(core_queues):
            for task in queue:
                core_exec_sum[core] += task.exec_time

    if print_out:
        logging.info("-"*20)
        for nc, c in enumerate(core_queues):
            # convert list to dict
            sorted_core_queue = sorted(c, key=lambda x: x.finish_time())
            logging.info("core %s: %s", nc, {t.id: (t.start_time, t.finish_time()) for t in sorted_core_queue})
        longest = 0
        for t in new_tasks:
            longest = t.finish_time() if t.finish_time() > longest else longest
        logging.info('makespan: %s', longest)
    return max(core_times), core_exec_sum


def get_individual_fitness(tasks, individual):
    """
    Determine the fitness of an individual. Higher is better.
    individual: the individual to evaluate
    """
    new_tasks = copy.deepcopy(tasks)

    core_queues = [[] for _ in range(total_cores)]
    core_times = [0 for _ in range(total_cores)]
    # for each scheduled task do this
    for index in range(len(new_tasks)):
        # find the targeted core
        core = individual[index]
        # append the task to the targeted core_queue
        core_queues[core].append(new_tasks[index])
        new_tasks[index].core = core
        # set the task's start_time to core's current time
        new_tasks[index].start_time = core_times[core] # + 1
        for dep in new_tasks[index].deps:
            # calculate end_time for task's dependencies
            dep_end_time = dep.start_time + dep.exec_time * (low_perf_multiplier if dep.core < total_cores / 2 else 1)
            # check if dep end_time is already passed
            if new_tasks[index].start_time < dep_end_time:
                new_tasks[index].start_time = dep_end_time
        exec_time = new_tasks[index].exec_time * (low_perf_multiplier if core < total_cores / 2 else 1)
        core_times[core] = new_tasks[index].start_time + exec_time

    score = 0
    missed = 0
    for index in range(len(core_queues)):
        qs, qm = fitness_for_queue(index, core_queues[index])
        score += qs
        missed += qm

    return score, missed


def get_population_fitness(tasks, population):
    scores = list()
    misses = list()
    for individual in population:
        score, miss = get_individual_fitness(tasks, individual)
        scores.append(score)
        misses.append(miss)
    return scores, misses


def grade(tasks, population):
    """
    Find average fitness for a population.
    """
    score = 0
    missed = 0
    for individual in population:
        ps, pm = get_individual_fitness(tasks, individual)
        score += ps
        missed += pm

    return score/len(population), missed/len(population)


def crossover(father, mother):
    gene_size = len(father)
    # 0 means father
    gene_pool = [0 for _ in range(gene_size)]
    goal = gene_size / 2
    changed = 0
    while changed < goal:
        position = randint(0, gene_size - 1)
        # we are taking this gene from mama
        if gene_pool[position] == 0:
            changed += 1
            gene_pool[position] = 1

    return [mother[index] if gene_pool[index] == 1 else father[index] for index in range(gene_size)]


def evolve(tasks, population, fitness, retain=0.2, random_select=0.05, mutate=0.15):
    graded = []
    for idx, individual in enumerate(population):
        # individual_score, _ = get_individual_fitness(tasks, individual)
        graded.append((individual, fitness[idx]))

    graded.sort(key=lambda tup: tup[1], reverse=True)
    retain_length = int(len(graded) * retain)
    parents = []
    for item in graded[:retain_length]:
        parents.append(item[0])
    # randomly add other individuals to
    # promote genetic diversity
    for item in graded[retain_length:]:
        if random() < random_select:
            parents.append(item[0])

    # mutate some individuals
    for individual in parents:
        if random() < mutate:
            position = randrange(0, len(individual))
            individual[position] = get_core_for_task(tasks[position], 0.75)

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []
    while len(children) < desired_length:
        father = randint(0, parents_length - 1)
        mother = randint(0, parents_length - 1)
        if father != mother:
            father = parents[father]
            mother = parents[mother]
            child = crossover(father, mother)
            children.append(child)
    parents.extend(children)
    return parents


def get_tasks_from_file(file_name):
    tg = set()
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            tg.add(line.split()[0])
    return tg


def parse_tasks(task_file='deadline.stg'):
    '''
    task graph file format:
    first line consists of only one value: number of nodes
    each of the following lines consists of tab separated values.
    id, exec_time, deadline, number of deps, list of dependencies
    '''
    tasks = []
    tg_1 = get_tasks_from_file(task_file)

    with open(task_file) as f:
        lines = f.readlines()
        size = int(lines[0])
        lines = lines[1:]
        for index in range(size):
            values = [int(x) for x in lines[index].split()]
            t = Task()
            t.id = values[0]
            t.exec_time = values[1]
            t.deadline = values[2]
            t.tg_id = 1 if t.id in tg_1 else 2
            for j in range(values[3]):
                t.deps.append(tasks[values[4 + j]])
            tasks.append(t)
    return tasks


def get_min_deadline(task):
    min_deadline = 0
    if len(task.deps) == 0:
        return task.exec_time

    for dep in task.deps:
        min_deadline = max(min_deadline, get_min_deadline(dep))

    return min_deadline + task.exec_time


def add_deadline(src, dst='deadline.stg'):
    '''
    Standard Task Graph Sets are obtained from:
    http://www.kasahara.elec.waseda.ac.jp/schedule/
    this function appends a deadline column to stg files
    '''
    tasks = []
    with open(src) as f:
        lines = f.readlines()
        size = int(lines[0])
        lines = lines[1:]
        for index in range(size):
            values = [int(x) for x in lines[index].split()]
            task = Task()
            task.id = values[0]
            task.exec_time = values[1]
            for j in range(values[2]):
                task.deps.append(tasks[values[3 + j]])
            tasks.append(task)

    with open(dst, 'w') as o:
        print(size, file=o)
        so_far = 0
        for task in tasks:
            task.deadline = int(1.5 * (max(get_min_deadline(task), so_far / 2)) + task.exec_time)
            so_far += task.exec_time

            print('{0: <6}'.format(task.id), end=' \t ', file=o)
            print('{0: <6}'.format(task.exec_time), end=' \t ', file=o)
            print('{0: <6}'.format(task.deadline), end=' \t ', file=o)
            print('{0: <6}'.format(len(task.deps)), end=' \t ', file=o)
            for d in task.deps:
                print('{0: <6}'.format(d.id), end=' \t ', file=o)
            print('', file=o)


def plot(history_max, history_min, history_avg, makespans, tot_generations):
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        print("Using matplotlib to show the fitness/generation plot...")
        array = np.arange(1, tot_generations+1, dtype='int32')

        fig, ax1 = plt.subplots()

        ax1.set_xlabel('generation (#)')
        ax1.set_ylabel('fitness')
        ax1.plot(array, history_avg, color='blue', marker='^', markersize=6, markevery=10, label='Mean')
        ax1.plot(array, history_min, color='yellow', marker='^', markersize=6, markevery=10, label='Min')
        ax1.plot(array, history_max, color='red', marker='^', markersize=6, markevery=10, label='Max')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
        #plt.xlim((0,tot_generations))
        #plt.ylim((-100,+100))
        # ax1.ylabel('Fitness', fontsize=15)
        # ax1.xlabel('Generation', fontsize=15)

        ax2 = ax1.twinx() # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('makespan (s)')
        ax2.plot(array, makespans, color='black', marker='^', markersize=6, markevery=10, label='Makespan')
        ax2.tick_params(axis='y')

        fig.tight_layout()  # otherwise the right y-label is slightly clipped

        print("Saving the image in './fitness.jpg'...")
        plt.savefig("./fitness.jpg", dpi=500)
        # plt.show()
    except ImportError:
        print("Please install matplotlib if you want to see the fitness/generation plot.")
        pass # module doesn't exist, deal with it. 




def main():
    # add_deadline(src='stg/fft.stg', dst='deadline.stg')
    # add_deadline(src='stg/laplace.stg', dst='deadline.stg')
    # add_deadline(src='stg/gaussian_elimination.stg', dst='deadline.stg')
    if task_graph in stgs:
        add_deadline(src='stg/' + task_graph, dst='deadline.stg')
    fitness_mean_history = list()
    fitness_min_history = list()
    fitness_max_history = list()
    makespan_history = list()
    tasks = parse_tasks()
    population_size = 200
    tot_generations = 50
    logging.info("population: {}  total generations: {}".format(population_size, tot_generations))
    logging.info("="*20 + " Tasks " + "="*20)
    for t in tasks:
        logging.info(t)
    logging.info("="*50)

    population = create_population(tasks, population_size)
    fitness, _ = get_population_fitness(tasks, population)

    # fitness_history = [grade(tasks, population), ]
    fitness_max_history.append( max(fitness) )
    fitness_min_history.append( min(fitness) )
    fitness_mean_history.append( sum(fitness)/len(fitness) )

    core_exec_sum = list()
    best = None

    for i in range(tot_generations):
        population = evolve(tasks, population, fitness)
        fitness, missed = get_population_fitness(tasks, population)
        
        fitness_max_history.append( max(fitness) )
        fitness_min_history.append( min(fitness) )
        fitness_mean_history.append( sum(fitness)/len(fitness) )
        # score, missed = grade(tasks, population)
        score = max(fitness)
        missed = sum(missed)/len(missed)
        makespan, core_exec_sum = get_makespan(tasks, population[maxl(fitness)])
        best = population[ fitness.index(score) ]
        # fitness_history.append(score)
        makespan_history.append(makespan)
        logging.info('iteration {} max_score: {} avg_missed: {} makespan: {}'.format(i + 1, score, missed, makespan))

    logging.info("totla execution time on each core: %s" % core_exec_sum)
    logging.info("best schedule: %s", best)
    get_makespan(tasks, best, True)
    
    plot(fitness_max_history[0:-1],
                fitness_min_history[0:-1], 
                fitness_mean_history[0:-1],
                makespan_history,
                tot_generations)




if __name__ == '__main__':
    main()
