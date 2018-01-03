"""
# Example usage
from genetic import *
target = 371
p_count = 100
i_length = 6
i_min = 0
i_max = 100
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]
for i in xrange(100):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))

for datum in fitness_history:
   print datum
"""
from __future__ import print_function

import copy
from random import randint, random, randrange

total_cores = 8
low_perf_multiplier = 2


class Task:
    def __init__(self):
        self.deps = []
        self.id = 0
        self.exec_time = 0
        self.deadline = 0
        self.start_time = 0
        self.core = -1

    def __repr__(self):
        return 'id: {} core: {} start_time: {} exec_time: {}'.format(
            self.id,
            self.core,
            self.start_time,
            self.exec_time
        )


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


def create_individual(tasks, low_percent=0.75):
    """
    Create a member of the population.

    """
    schedule = [0 for _ in range(len(tasks))]
    for task in tasks:
        if random() < low_percent:
            selected_core = randrange(total_cores / 2)
        else:
            selected_core = randrange(total_cores / 2, total_cores)
        schedule[task.id] = selected_core

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
    for task in queue:
        exec_time = task.exec_time * (low_perf_multiplier if core < total_cores / 2 else 1)
        if task.deadline > task.start_time + exec_time:
            score -= 1000
        else:
            score += task.deadline - task.start_time - exec_time
    return score


def fitness(tasks, individual):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    """
    new_tasks = copy.deepcopy(tasks)

    core_queues = [[] for _ in range(total_cores)]
    core_times = [0 for _ in range(total_cores)]
    for index in range(len(new_tasks)):
        core = individual[index]
        core_queues[core].append(new_tasks[index])
        new_tasks[index].core = core
        new_tasks[index].start_time = core_times[core] + 0
        exec_time = new_tasks[index].exec_time * (low_perf_multiplier if core < total_cores / 2 else 1)
        core_times[core] += exec_time

    for task in new_tasks:
        for dep in task.deps:
            dep_end_time = dep.start_time + dep.exec_time * (low_perf_multiplier if dep.core < total_cores / 2 else 1)
            if task.start_time < dep_end_time:
                task.start_time = dep_end_time

    score = 0
    for index in range(len(core_queues)):
        score += fitness_for_queue(index, core_queues[index])

    return score


def grade(tasks, population):
    """
    Find average fitness for a population.

    """
    summed = 0
    for individual in population:
        score = fitness(tasks, individual)
        summed += score
    return summed


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

    return [mother[index] if gene_pool == 1 else father[index] for index in range(gene_size)]


def evolve(tasks, population, retain=0.1, random_select=0.05, mutate=0.2):
    graded = []
    for individual in population:
        graded.append((individual, fitness(tasks, individual)))

    graded.sort(key=lambda tup: tup[1], reverse=True)
    retain_length = int(len(graded) * retain)
    parents = []
    for item in graded[:retain_length]:
        parents.append(item[0])
    # randomly add other individuals to
    # promote genetic diversity
    for item in graded[retain_length:]:
        if random_select > random():
            parents.append(item[0])

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            position = randrange(0, len(individual))
            individual[position] = randrange(0, total_cores)  # todo: fix this

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


def parse_tasks():
    tasks = []
    with open('deadline.stg') as f:
        lines = f.readlines()
        size = int(lines[0])
        lines = lines[1:]
        for index in range(size + 2):
            values = [int(x) for x in lines[index].split()]
            t = Task()
            t.id = values[0]
            t.exec_time = values[1]
            t.deadline = values[2]
            for j in range(values[3]):
                t.deps.append(tasks[values[4 + j]])
            tasks.append(t)
    return tasks


def add_deadline():
    tasks = []
    with open('robot.stg') as f:
        lines = f.readlines()
        size = int(lines[0])
        lines = lines[1:]
        for index in range(size + 2):
            values = [int(x) for x in lines[index].split()]
            t = Task()
            t.id = values[0]
            t.exec_time = values[1]
            for j in range(values[2]):
                t.deps.append(tasks[values[3 + j]])
            tasks.append(t)

    with open('deadline.stg', 'w') as o:
        print(size, file=o)
        so_far = 0
        for task in tasks:
            if task.deps:
                task.deadline = so_far / 3 + task.exec_time
            else:
                task.deadline = 2 * task.exec_time
            so_far += task.exec_time

            print('{0: <6}'.format(task.id), end=' \t ', file=o)
            print('{0: <6}'.format(task.exec_time), end=' \t ', file=o)
            print('{0: <6}'.format(task.deadline), end=' \t ', file=o)
            print('{0: <6}'.format(len(task.deps)), end=' \t ', file=o)
            for d in task.deps:
                print('{0: <6}'.format(d.id), end=' \t ', file=o)
            print('', file=o)


def main():
    # add_deadline()
    tasks = parse_tasks()
    population_size = 200
    population = create_population(tasks, population_size)
    fitness_history = [grade(tasks, population), ]
    for i in range(250):
        population = evolve(tasks, population)
        g = grade(tasks, population)
        fitness_history.append(g)
        print('iteration {} score: {}'.format(i + 1, g))


if __name__ == '__main__':
    main()
