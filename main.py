import random
import math
import timeit
import matplotlib.pyplot as plt
from numpy.random import choice

N = 3
sols_per_pop = 200
num_gens = 200
num_parents_mating = 22
MUTATION_LIMIT = 1
THRESHOLD_FITNESS = 0.002

test_pts = []
x = 0.1
while x <= 10:
	test_pts.append(x)
	x += 0.01

def real(x):
    return 1 / x

def f(x, coeff):
    val = (coeff[0] + coeff[1] * x + coeff[2] / x) * (1 / (x * (x + 1) * (x + 5)))
    return val

def plot(sol):
    #print(test_pts)
    x = test_pts
    y = [f(i, sol) for i in x]
    # plt.xticks(range(1, 11))
    # plt.yticks(range(1, 11))
    figure, axis = plt.subplots(1,3)
    axis[0].plot(x, y, label="Using Genetic algoritm")
    axis[0].plot(x, [real(i) for i in x],  label="Original step function")
    axis[0].legend(loc='best')
    axis[0].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    axis[1].plot(x, y, label="Using Genetic algoritm")
    axis[1].legend(loc='best')
    axis[1].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    axis[2].plot(x, [real(i) for i in x],  label="Original step function")
    axis[2].legend(loc='best')
    axis[2].grid(color = 'green', linestyle = '--', linewidth = 0.5)
    plt.show()

def generate_initial_population():
    population = []
    for i in range(sols_per_pop):
        sol =  random.sample(range(1, 300), N)
        population.append(sol)
    return population

def calc_sol_perc_error(sol):
    error = 0
    for pt in test_pts:
        error += abs((real(pt) - f(pt, sol)) / real(pt))
    return 100 * error / len(test_pts)

def calc_sol_fitness(sol):
    error = 0
    for pt in test_pts:
        error += (real(pt) - f(pt, sol)) ** 2
    return error / len(test_pts)

def calc_pop_fitness(population):
    fitness = []
    for sol in population:
        fitness.append(calc_sol_fitness(sol))
    return fitness

def select_mating_pool_by_fittest(pop, fitness, num_parents):
    parents = []
    fitness_copy = fitness.copy()
    for i in range(num_parents):
        idx_of_parent = fitness_copy.index(min(fitness_copy))
        parents.append(pop[idx_of_parent])
        fitness_copy[idx_of_parent] = 999999999
    return parents

def get_prob_without_ranking(fitness):
    fitness_sum = sum(fitness)
    inv_probs = [f / fitness_sum for f in fitness]
    probs = [(1.0 - p) for p in inv_probs]
    probs = [p / sum(probs) for p in probs]
    return probs

def get_prob_with_ranking(fitness):
    tmp = sorted(fitness, reverse=True)
    probs_tmp = [tmp.index(x) for x in fitness]
    probs_sum = sum(probs_tmp)
    probs = [p / probs_sum for p in probs_tmp]
    return probs


def select_mating_pool_by_roulette(pop, fitness, num_parents):
    ret_idx = set()
    probs = get_prob_with_ranking(fitness)
    while len(ret_idx) != num_parents:
        idx = choice(range(len(probs)), p = probs)
        if idx not in ret_idx:
            ret_idx.add(idx)
    return [pop[idx] for idx in ret_idx]

def generate_offspring_random_biased(parent1, parent2):
    offspring = []
    p1f = calc_sol_fitness(parent1)
    p2f = calc_sol_fitness(parent2)
    if p1f < p2f:
        for i in range(len(parent1)):
            if random.random() < 0.7:
                offspring.append(parent1[i])
            else:
                offspring.append(parent2[i])       
    else:
        for i in range(len(parent1)):
            if random.random() < 0.7:
                offspring.append(parent2[i])
            else:
                offspring.append(parent1[i])
        
    return offspring

def crossover(parents, num_offsprings):
    offprings = []
    for k in range(num_offsprings):
        parent_1 = parents[k % len(parents)]
        parent_2 = parents[(k + 1) % len(parents)]
        child = generate_offspring_random_biased(parent_1, parent_2)
        offprings.append(child)
    return offprings

def mutate_offspring_gd(offspring):
    rand_idx = random.randrange(len(offspring))
    rand_val = random.uniform(0, MUTATION_LIMIT)
    offspring[rand_idx] += rand_val
    err1 = calc_sol_fitness(offspring)
    offspring[rand_idx] -= 2 * rand_val
    err2 = calc_sol_fitness(offspring)
    if(err1 < err2):
        offspring[rand_idx] += 2 * rand_val
    return offspring
                           
def mutation(offspring_crossover):
    mutated_offsprings = []
    for offspring in offspring_crossover:
        mutated_offsprings.append(mutate_offspring_gd(offspring))
    return mutated_offsprings

def best_solution_in_population(population):
    mn = 10000000000
    ret = None  
    for sol in population:
        if calc_sol_fitness(sol) < mn:
            mn = calc_sol_fitness(sol)
            ret = sol
    return ret

def add_offsprings(population, offspring_mutation, fitness):
    idx_to_chop = []
    for i in range(len(offspring_mutation)):
        idx_of_max = fitness.index(max(fitness))
        idx_to_chop.append(idx_of_max)
        fitness[idx_of_max] = -1000000000

    for i in range(len(offspring_mutation)):
        population[idx_to_chop[i]] = offspring_mutation[i]
    return population 


def add_offsprings_with_prob(population, offspring_mutation, fitness):
    tmp = sorted(fitness)
    bb = [tmp.index(f) for f in fitness]
    bb = [i+1 for i in bb]
    probs = [x/sum(bb) for x in bb]
    indexes = []
    while len(indexes) != len(offspring_mutation):
        idx = choice(range(len(probs)), p=probs)
        if idx not in indexes:
            indexes.append(idx)
    for i in range(len(offspring_mutation)):
        idx = indexes[i]
        population[idx] = offspring_mutation[i]
    return population

def main():
    start = timeit.default_timer()
    global MUTATION_LIMIT
    population = generate_initial_population()
    print("Initial solution:", best_solution_in_population(population))

    for gen in range(num_gens):
        fitness = calc_pop_fitness(population)
        min_fitness = min(fitness)
        print("gen={g}, min_fitness={mf}".format(g=gen, mf=min_fitness))
        parents = select_mating_pool_by_roulette(population, fitness, num_parents_mating)
        offspring_crossover = crossover(parents, num_offsprings = len(population) - len(parents))
        offspring_mutation = mutation(offspring_crossover)
        population = add_offsprings_with_prob(population, offspring_mutation, fitness)  
    
    coeff = best_solution_in_population(population)

    print("Best solution:", coeff)
    print("Mean percentage error:", calc_sol_perc_error(coeff))
    
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    plot(coeff)

if __name__ == '__main__':
    main()