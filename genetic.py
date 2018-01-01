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

from random import randint, random, randrange

import numpy as np

core_count = 8


class Task:
    def __init__(self):
        self.deps = []
        self.id = 0
        self.exec_time = 0
        self.deadline = 0


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


def get_random_core(cores, low=True):
    return randrange(cores / 2) if low else randrange(cores / 2, cores)


def can_schedule_on_low(core_queues, task):
    for queue in core_queues:
        if queue.low:
            if task.deadline <= queue.last_job + task.exec_time * 2:
                return True
    return False


def create_individual(tasks, cores=core_count):
    """
    Create a member of the population.

    """
    core_queues = [CoreQueue() for _ in range(cores)]
    schedule = np.zeros((2, len(tasks)))
    for task in tasks:
        job = Job(task.id)
        if can_schedule_on_low(core_queues, task):
            core = get_random_core(cores)
        else:
            core = get_random_core(cores, False)
        job.start = core_queues[core].last_job

        if core > cores / 2:
            job.end = job.start + task.exec_time
            schedule[0][task.id] = core
        else:
            job.end = job.start + 2 * task.exec_time
            schedule[1][task.id] = core
        core_queues[core].jobs.append(job)
        core_queues[core].last_job = job.end

    return schedule


def create_population(count, cores=core_count):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [create_individual(cores) for _ in range(count)]


def fitness_for_queue(core, queue):
    score = 0
    time = 0
    missed = 0
    for item in queue:
        exec_time_on_core = item.exec_time if core < core_count / 2 else item.exec_time * 2
        if time + exec_time_on_core > item.deadline:
            score -= 100000
            missed += 1
        else:
            score += item.deadline - time - exec_time_on_core
        time += exec_time_on_core

    return score, missed


def fitness(individual):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    """
    core_queues = [[] for _ in range(core_count)]
    for index in range(len(tasks)):
        core_queues[individual[index]].append(tasks[index])

    score = 0
    missed = 0
    for index in range(len(core_queues)):
        qs, qm = fitness_for_queue(index, core_queues[index])
        score += qs
        missed += qm

    return score, missed


def grade(population):
    """
    Find average fitness for a population.

    """
    summed = 0
    total_misses = 0
    perfect_individuals = 0
    for individual in population:
        score, missed = fitness(individual)
        summed += score
        total_misses += missed
        if missed == 0:
            perfect_individuals += 1
    return summed / (len(population) * 1.0), total_misses, perfect_individuals


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


def evolve(population, retain=0.15, random_select=0.05, mutate=0.02):
    graded = [(fitness(x), x) for x in population]
    graded = [x[1] for x in sorted(graded)]
    graded = graded[::-1]
    retain_length = int(len(graded) * retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)

    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual) - 1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(0, core_count - 1)

    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(population) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length - 1)
        female = randint(0, parents_length - 1)
        if male != female:
            male = parents[male]
            female = parents[female]
            child = crossover(male, female)
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
                task.deadline = so_far / 4 + task.exec_time
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


if __name__ == '__main__':
    # add_deadline()
    tasks = parse_tasks()
    create_individual(tasks)
    # population_count = 1000
    # p = create_population(population_count)
    # fitness_history = [grade(p), ]
    # for i in range(1000):
    #     p = evolve(p)
    #     g, m, pi = grade(p)
    #     fitness_history.append((g, m, pi))
    #     print(i + 1, g, m, pi)
